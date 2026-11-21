"""User data: accounts, sessions, onboarding prefs, diary, feedback, agent traces. SQLite, per-user isolation by user_id on every query."""
import threading, sqlite3, os, secrets, time, json
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
PH = PasswordHasher()
DATA = os.environ.get('ODDLY_DATA', os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
SCHEMA = '''
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT, pw_hash TEXT, google_sub TEXT UNIQUE, created REAL);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS prefs(user_id INT PRIMARY KEY, genres TEXT DEFAULT '[]', picks TEXT DEFAULT '[]', services TEXT DEFAULT '[]', languages TEXT DEFAULT '[]', onboarded INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS diary(user_id INT, title_id TEXT, status TEXT CHECK(status IN('watched','want','dropped')), rating REAL, note TEXT DEFAULT '', updated REAL, PRIMARY KEY(user_id,title_id));
CREATE TABLE IF NOT EXISTS ott_cache(title_id TEXT PRIMARY KEY, fetched REAL, payload TEXT);
CREATE TABLE IF NOT EXISTS quota(k TEXT PRIMARY KEY, n INT NOT NULL, exp REAL NOT NULL);
CREATE TABLE IF NOT EXISTS ott_budget(day TEXT PRIMARY KEY, n INT);
CREATE TABLE IF NOT EXISTS ott_err(title_id TEXT PRIMARY KEY, fetched REAL);
CREATE TABLE IF NOT EXISTS feedback(user_id INT, title_id TEXT, kind TEXT, reason TEXT DEFAULT '', created REAL, PRIMARY KEY(user_id,title_id,kind));
CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY, user_id INT, run_id TEXT, node TEXT, started REAL, ms REAL, detail TEXT, llm_tokens INT);
'''
class Store:
    def __init__(self, path=None):
        self.lock = threading.RLock()
        self.path = path or os.path.join(DATA, 'users.sqlite')
        if os.environ.get('TURSO_DATABASE_URL') and os.environ.get('TURSO_AUTH_TOKEN') and not path:
            from .tursodb import TursoConn, TursoError
            self.c = TursoConn(os.environ['TURSO_DATABASE_URL'], os.environ['TURSO_AUTH_TOKEN']); self.backend = 'turso'; ALTER_ERR = TursoError
        else:
            self.c = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=15); self.c.row_factory = sqlite3.Row; self.backend = 'sqlite'; ALTER_ERR = sqlite3.OperationalError
        self.c.executescript(SCHEMA)
        try: self.c.execute('alter table prefs add column spoiler_safe INT DEFAULT 0')
        except ALTER_ERR: pass
        self.c.commit()
    def _q(self, sql, a=()):
        with self.lock: return self.c.execute(sql, a)
    def _tx(self, stmts):
        """All-or-nothing group of writes. Returns last_insert_rowid of the first statement."""
        if self.backend == 'turso': return self.c.transaction(stmts)
        with self.lock:
            self.c.execute('BEGIN IMMEDIATE')
            try:
                first = None
                for i, (sql, a) in enumerate(stmts):
                    cur = self.c.execute(sql, a)
                    if i == 0: first = cur.lastrowid
                self.c.execute('COMMIT'); return first
            except Exception:
                self.c.execute('ROLLBACK'); raise
    def quota_take(self, key, limit, ttl):
        """Durable atomic counter: one statement increments only while n < limit (so concurrent callers cannot overshoot). True if a slot was taken.
        Keys are opaque (callers pass hashed identifiers, never raw IPs); rows expire after ttl seconds and are purged opportunistically."""
        now = time.time()
        with self.lock:
            if int(now * 1000) % 20 == 0: self.c.execute('delete from quota where exp<?', (now,)); self.c.commit()
            row = self.c.execute('insert into quota(k,n,exp) values(?,1,?) on conflict(k) do update set n=n+1 where n<? returning n', (key, now + ttl, limit)).fetchone()
            self.c.commit()
        return row is not None
    def create_user(self, email, name, password=None, google_sub=None):
        email = email.strip().lower()
        if self._q('select 1 from users where email=?', (email,)).fetchone(): raise ValueError('email already registered')
        return self._tx([('insert into users(email,name,pw_hash,google_sub,created) values(?,?,?,?,?)', (email, name, PH.hash(password) if password else None, google_sub, time.time())),
                         ('insert into prefs(user_id) values(last_insert_rowid())', ())])
    def check_password(self, email, password):
        r = self._q('select * from users where email=?', (email.strip().lower(),)).fetchone()
        if not r or not r['pw_hash']: return None
        try: PH.verify(r['pw_hash'], password); return r['id']
        except VerifyMismatchError: return None
    def google_user(self, sub, email, name):
        r = self._q('select id from users where google_sub=?', (sub,)).fetchone()
        if r: return r['id']
        r = self._q('select id from users where email=?', (email.lower(),)).fetchone()
        if r: self._q('update users set google_sub=? where id=?', (sub, r['id'])); self.c.commit(); return r['id']
        return self.create_user(email, name, google_sub=sub)
    def new_session(self, uid):
        t = secrets.token_urlsafe(32); self._q('insert into sessions values(?,?,?)', (t, uid, time.time())); self.c.commit(); return t
    def user_for(self, token):
        if not token: return None
        r = self._q('select u.id,u.email,u.name from sessions s join users u on u.id=s.user_id where s.token=?', (token,)).fetchone()
        return dict(r) if r else None
    def delete_user(self, uid):
        self._tx([(f'delete from {t} where user_id=?', (uid,)) for t in ('sessions', 'prefs', 'diary', 'feedback', 'traces')] + [('delete from users where id=?', (uid,))])
    def end_session(self, token): self._q('delete from sessions where token=?', (token,)); self.c.commit()
    def prefs(self, uid):
        r = self._q('select * from prefs where user_id=?', (uid,)).fetchone()
        return {'genres': json.loads(r['genres']), 'picks': json.loads(r['picks']), 'services': json.loads(r['services']), 'languages': json.loads(r['languages']), 'onboarded': bool(r['onboarded']), 'spoiler_safe': bool(r['spoiler_safe'])}
    def save_prefs(self, uid, **kw):
        if not kw: return
        self._tx([(f'update prefs set {k}=? where user_id=?', (int(v) if k in ('onboarded', 'spoiler_safe') else json.dumps(v), uid)) for k, v in kw.items()])
    def set_diary(self, uid, tid, status, rating=None, note=''):
        self._q('insert into diary values(?,?,?,?,?,?) on conflict(user_id,title_id) do update set status=excluded.status, rating=excluded.rating, note=excluded.note, updated=excluded.updated', (uid, tid, status, rating, note, time.time())); self.c.commit()
    def del_diary(self, uid, tid): self._q('delete from diary where user_id=? and title_id=?', (uid, tid)); self.c.commit()
    def diary(self, uid): return [dict(r) for r in self._q('select title_id,status,rating,note,updated from diary where user_id=? order by updated desc', (uid,))]
    def add_feedback(self, uid, tid, kind, reason=''):
        self._q('insert or replace into feedback values(?,?,?,?,?)', (uid, tid, kind, reason, time.time())); self.c.commit()
    def feedback(self, uid, kind='not_for_me'): return [r['title_id'] for r in self._q('select title_id from feedback where user_id=? and kind=?', (uid, kind))]
    def trace(self, uid, run_id, node, started, ms, detail, tokens=None):
        self._q('insert into traces(user_id,run_id,node,started,ms,detail,llm_tokens) values(?,?,?,?,?,?,?)', (uid, run_id, node, started, ms, json.dumps(detail)[:2000], tokens)); self.c.commit()
    def trace_stats(self):
        rows = self._q('select node,count(*) n,avg(ms) avg_ms,max(ms) max_ms from traces group by node').fetchall()
        runs = self._q('select count(distinct run_id) from traces').fetchone()[0]
        return {'runs': runs, 'nodes': [dict(r) for r in rows], 'llm_tokens': (self._q('select sum(llm_tokens) from traces').fetchone()[0])}
    def run_trace(self, uid, run_id): return [dict(r) for r in self._q('select node,ms,detail from traces where user_id=? and run_id=? order by id', (uid, run_id))]
