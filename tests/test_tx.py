"""Atomicity and persistence checks for Store. The Turso path is exercised against a local fake of Hrana's pipeline/batch semantics
(conditions evaluated per step), so it tests our request shape and rollback handling, not Turso itself. Real-service checks are done on the deployment."""
import json, sqlite3, threading, httpx, pytest
from oddly.store import Store
from oddly.tursodb import TursoConn, TursoError

def fake_hrana(db):
    lock = threading.Lock()
    def cols_rows(cur):
        names = [d[0] for d in cur.description] if cur.description else []
        def enc(v): return {'type': 'null'} if v is None else {'type': 'integer', 'value': str(v)} if isinstance(v, int) else {'type': 'float', 'value': v} if isinstance(v, float) else {'type': 'text', 'value': v}
        return {'cols': [{'name': n} for n in names], 'rows': [[enc(v) for v in r] for r in cur.fetchall()] if names else [], 'affected_row_count': max(cur.rowcount, 0), 'last_insert_rowid': str(cur.lastrowid) if cur.lastrowid else None}
    def run(stmt):
        args = [a.get('value') if a['type'] != 'integer' else int(a['value']) for a in stmt.get('args', [])]
        return cols_rows(db.execute(stmt['sql'], args))
    def handler(req):
        body = json.loads(req.content); out = []
        with lock:
            for r in body['requests']:
                if r['type'] == 'close': out.append({'type': 'ok', 'response': {'type': 'close'}}); continue
                if r['type'] == 'execute':
                    try: out.append({'type': 'ok', 'response': {'type': 'execute', 'result': run(r['stmt'])}})
                    except sqlite3.Error as e: out.append({'type': 'error', 'error': {'message': str(e)}})
                else:
                    res, errs = [], []
                    def ok(c):
                        if c['type'] == 'ok': return res[c['step']] is not None
                        if c['type'] == 'not': return not ok(c['cond'])
                        return True
                    for st in r['batch']['steps']:
                        if 'condition' in st and not ok(st['condition']): res.append(None); errs.append(None); continue
                        try: res.append(run(st['stmt'])); errs.append(None)
                        except sqlite3.Error as e: res.append(None); errs.append({'message': str(e)})
                    out.append({'type': 'ok', 'response': {'type': 'batch', 'result': {'step_results': res, 'step_errors': errs}}})
        return httpx.Response(200, json={'results': out})
    return handler

def make_turso_store():
    db = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
    client = httpx.Client(transport=httpx.MockTransport(fake_hrana(db)))
    s = Store.__new__(Store); s.c = TursoConn('libsql://x.invalid', 't', client=client); s.backend = 'turso'; s.lock = threading.RLock()
    from oddly.store import SCHEMA; s.c.executescript(SCHEMA); s.c.execute('alter table prefs add column spoiler_safe INT DEFAULT 0'); return s, db

@pytest.fixture(params=['sqlite', 'turso'])
def store(request, tmp_path):
    if request.param == 'sqlite': return Store(str(tmp_path / 'u.sqlite'))
    return make_turso_store()[0]

def test_signup_creates_user_and_prefs_together(store):
    uid = store.create_user('a@x.io', 'A', 'pw-123456789')
    assert store.prefs(uid)['onboarded'] is False and store.check_password('a@x.io', 'pw-123456789') == uid

def test_failed_group_rolls_back(store):
    uid = store.create_user('b@x.io', 'B', 'pw-123456789'); store.set_diary(uid, 'm1', 'watched', 4.0, 'n')
    with pytest.raises(Exception):
        store._tx([('delete from diary where user_id=?', (uid,)), ('delete from no_such_table', ())])
    assert len(store.diary(uid)) == 1   # first delete was rolled back, no partial deletion

def test_delete_user_is_all_or_nothing_and_complete(store):
    uid = store.create_user('c@x.io', 'C', 'pw-123456789'); tok = store.new_session(uid); store.set_diary(uid, 'm1', 'want'); store.add_feedback(uid, 'm2', 'not_for_me')
    store.delete_user(uid)
    assert store.user_for(tok) is None and store.diary(uid) == [] and store.feedback(uid) == []

def test_concurrent_diary_writes_keep_last_value_per_title(store):
    uid = store.create_user('d@x.io', 'D', 'pw-123456789')
    def w(i): store.set_diary(uid, f'm{i % 3}', 'watched', float(i % 5))
    ts = [threading.Thread(target=w, args=(i,)) for i in range(12)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert {d['title_id'] for d in store.diary(uid)} == {'m0', 'm1', 'm2'}

def test_duplicate_email_rejected(store):
    store.create_user('e@x.io', 'E', 'pw-123456789')
    with pytest.raises(ValueError): store.create_user('E@x.io', 'E2', 'pw-123456789')
