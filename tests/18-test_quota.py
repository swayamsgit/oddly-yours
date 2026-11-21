import threading, os, tempfile
from oddly.store import Store
from oddly import llm_agent

def test_quota_atomic_cap():
    s = Store(os.path.join(tempfile.mkdtemp(), 'q.sqlite')); got = []
    ts = [threading.Thread(target=lambda: got.append(s.quota_take('k', 5, 60))) for _ in range(20)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(got) == 5
    assert s.quota_take('other', 1, 60) and not s.quota_take('other', 1, 60)

def test_gemini_guard_blocks_when_counter_fails_or_full():
    old = llm_agent.QUOTA
    try:
        llm_agent.QUOTA = lambda k, l, t: False
        assert not llm_agent._gem_slot('m')
        def boom(*a): raise RuntimeError('db down')
        llm_agent.QUOTA = boom
        assert not llm_agent._gem_slot('m')
    finally: llm_agent.QUOTA = old

def test_signup_throttle_per_ip_and_no_raw_ip_stored():
    import oddly.server as srv
    from starlette.requests import Request
    def rq(ip): return Request({'type': 'http', 'headers': [(b'x-forwarded-for', ip.encode())], 'client': ('10.0.0.1', 1)})
    from fastapi import HTTPException
    ip = '203.0.113.%d' % (os.getpid() % 200)
    import time; srv.store.c.execute('delete from quota'); srv.store.c.commit()
    for _ in range(5): srv._throttle(rq(ip), 'signup', 5, 300)
    try: srv._throttle(rq(ip), 'signup', 5, 300); assert False
    except HTTPException as e: assert e.status_code == 429
    srv._throttle(rq('198.51.100.7'), 'signup', 5, 300)  # a different network is unaffected
    keys = ' '.join(r[0] for r in srv.store.c.execute('select k from quota').fetchall())
    assert ip not in keys and '198.51.100.7' not in keys

def test_poster_cap_is_atomic_under_concurrency(monkeypatch):
    from oddly import mpdb
    import io, json, threading
    monkeypatch.setenv('MOVIEPOSTERDB_API_KEY', 'test-key'); monkeypatch.setattr(mpdb, 'DAILY_CAP', 4)
    s = Store(os.path.join(tempfile.mkdtemp(), 'm.sqlite')); calls = []; lk = threading.Lock()
    def opener(rq, timeout=0):
        with lk: calls.append(1)
        return io.BytesIO(json.dumps({'primary_poster': {'url': 'https://posters.movieposterdb.com/x/s_1.jpg'}, 'url': 'https://movieposterdb.com/x'}).encode())
    res = []
    ts = [threading.Thread(target=lambda i=i: res.append(mpdb.lookup(s, 't%d' % i, 'tt%07d' % i, opener)['state'])) for i in range(16)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(calls) == 4 and res.count('budget') == 12   # fake provider only; no real calls
