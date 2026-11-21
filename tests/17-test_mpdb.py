import io, json, os
from oddly import mpdb
class R(io.BytesIO):
    def __enter__(s): return s
    def __exit__(s,*a): pass
def test_mpdb_strips_key_caps_and_caches(tmp_path, monkeypatch):
    from oddly.store import Store
    st = Store(str(tmp_path / 'u.sqlite'))
    monkeypatch.setenv('MOVIEPOSTERDB_API_KEY', 'mpdb_pk_secret')
    calls = []
    def op(rq, timeout=0):
        calls.append(rq.full_url)
        assert 'size=s' in rq.full_url and 'secret' not in rq.full_url
        return R(json.dumps({'url': 'https://movieposterdb.com/x', 'primary_poster': {'url': 'https://posters.movieposterdb.com/a/s_1_ab.jpg?k=mpdb_pk_secret'}}).encode())
    a = mpdb.lookup(st, 'm1', 'tt1', op); b = mpdb.lookup(st, 'm1', 'tt1', op)
    assert a['url'] == 'https://posters.movieposterdb.com/a/s_1_ab.jpg' and 'secret' not in json.dumps(a)
    assert b['cached'] and len(calls) == 1
    monkeypatch.setattr(mpdb, 'DAILY_CAP', 1)
    assert mpdb.lookup(st, 'm2', 'tt2', op)['state'] == 'budget'
    monkeypatch.delenv('MOVIEPOSTERDB_API_KEY'); assert mpdb.lookup(st, 'm3', 'tt3', op)['state'] == 'not_connected'

def test_search_empty_kind_means_all(tmp_path):
    import os
    from fastapi.testclient import TestClient
    from oddly.server import app
    c = TestClient(app)
    assert len(c.get('/api/search?q=the matrix&kind=&limit=36').json()) > 0
