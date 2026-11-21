"""Minimal Turso (libSQL) client over the documented HTTP pipeline API (Hrana over HTTP).
Exposes the small slice of the sqlite3 connection interface that oddly.store uses, so the same Store code runs locally on SQLite
and on the hosted build against Turso. Each statement is one HTTPS request and auto-commits; there are no multi-statement transactions."""
import base64, httpx

class _Row(dict):
    def __getitem__(self, k):
        return list(self.values())[k] if isinstance(k, int) else dict.__getitem__(self, k)

class _Cur:
    def __init__(self, rows, lastrowid, rowcount): self._rows, self.lastrowid, self.rowcount = rows, lastrowid, rowcount
    def fetchone(self): return self._rows[0] if self._rows else None
    def fetchall(self): return list(self._rows)
    def __iter__(self): return iter(self._rows)

def _arg(v):
    if v is None: return {'type': 'null'}
    if isinstance(v, bool): return {'type': 'integer', 'value': str(int(v))}
    if isinstance(v, int): return {'type': 'integer', 'value': str(v)}
    if isinstance(v, float): return {'type': 'float', 'value': v}
    if isinstance(v, (bytes, bytearray)): return {'type': 'blob', 'base64': base64.b64encode(bytes(v)).decode()}
    return {'type': 'text', 'value': str(v)}

def _val(c):
    t = c.get('type')
    if t == 'null': return None
    if t == 'integer': return int(c['value'])
    if t == 'float': return float(c['value'])
    if t == 'blob': return base64.b64decode(c['base64'])
    return c['value']

class TursoError(RuntimeError): pass

class TursoConn:
    def __init__(self, url, token, client=None):
        self.base = url.replace('libsql://', 'https://').rstrip('/')
        self.http = client or httpx.Client(timeout=20, headers={'Authorization': f'Bearer {token}'})
    def execute(self, sql, args=()):
        body = {'requests': [{'type': 'execute', 'stmt': {'sql': sql, 'args': [_arg(a) for a in args]}}, {'type': 'close'}]}
        r = self.http.post(self.base + '/v2/pipeline', json=body)
        if r.status_code != 200: raise TursoError(f'turso http {r.status_code}')
        res = r.json()['results'][0]
        if res.get('type') == 'error': raise TursoError(res['error'].get('message', 'turso error'))
        out = res['response']['result']
        names = [c['name'] for c in out['cols']]
        rows = [_Row(zip(names, [_val(c) for c in row])) for row in out['rows']]
        lr = out.get('last_insert_rowid')
        return _Cur(rows, int(lr) if lr is not None else None, out.get('affected_row_count', 0))
    def transaction(self, stmts):
        """Run [(sql, args), ...] atomically in one HTTP request using Hrana's documented batch with conditions:
        BEGIN, each statement only if the previous step succeeded, COMMIT only if all succeeded, ROLLBACK otherwise.
        Returns the last_insert_rowid of the first statement. Raises TursoError and leaves no partial writes."""
        steps = [{'stmt': {'sql': 'BEGIN IMMEDIATE'}}]
        for i, (sql, args) in enumerate(stmts):
            steps.append({'stmt': {'sql': sql, 'args': [_arg(a) for a in args]}, 'condition': {'type': 'ok', 'step': i}})
        n = len(stmts)
        steps.append({'stmt': {'sql': 'COMMIT'}, 'condition': {'type': 'ok', 'step': n}})
        steps.append({'stmt': {'sql': 'ROLLBACK'}, 'condition': {'type': 'not', 'cond': {'type': 'ok', 'step': n + 1}}})
        r = self.http.post(self.base + '/v2/pipeline', json={'requests': [{'type': 'batch', 'batch': {'steps': steps}}, {'type': 'close'}]})
        if r.status_code != 200: raise TursoError(f'turso http {r.status_code}')
        res = r.json()['results'][0]
        if res.get('type') == 'error': raise TursoError(res['error'].get('message', 'turso error'))
        br = res['response']['result']
        errs = [e for e in br['step_errors'] if e]
        if errs or br['step_results'][n + 1] is None: raise TursoError((errs[0] or {}).get('message', 'transaction failed and was rolled back') if errs else 'transaction failed and was rolled back')
        lr = br['step_results'][1].get('last_insert_rowid') if n else None
        return int(lr) if lr is not None else None
    def executescript(self, script):
        for s in [x.strip() for x in script.split(';') if x.strip()]: self.execute(s)
    def commit(self): pass
