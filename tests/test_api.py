import os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
SRC = os.environ.get('ODDLY_DATA', '/tmp/oddly/data'); tmp = tempfile.mkdtemp()
for f in ['catalogue.sqlite', 'emb.npy', 'emb_ids.json', 'item_knn.npz', 'item_factors.npy', 'eval_results.json']: shutil.copy(os.path.join(SRC, f), tmp)
os.environ['ODDLY_DATA'] = tmp
from fastapi.testclient import TestClient
from oddly.server import app
from oddly.agent import parse_like_but
def user(email):
    c = TestClient(app); assert c.post('/api/auth/signup', json={'email': email, 'name': 'T', 'password': 'longenough1'}).status_code == 200; return c
def test_parse_like_but():
    p = parse_like_but('like Interstellar, but under two hours'); assert p['seed'] == 'interstellar' and p['constraints']['max_runtime'] == 120
    assert parse_like_but('like Parasite but in hindi')['constraints']['language'] == 'Hindi'
def test_auth_required_and_isolation():
    assert TestClient(app).get('/api/me').status_code == 401
    a, b = user('a@t.io'), user('b@t.io'); tid = a.get('/api/search?q=Inception').json()[0]['id']
    assert a.put('/api/diary', json={'title_id': tid, 'status': 'watched', 'rating': 4}).status_code == 200
    assert len(a.get('/api/diary').json()) == 1 and b.get('/api/diary').json() == []
def test_wrong_password_and_duplicate():
    c = user('c@t.io'); assert TestClient(app).post('/api/auth/login', json={'email': 'c@t.io', 'password': 'nope'}).status_code == 401
    assert TestClient(app).post('/api/auth/signup', json={'email': 'c@t.io', 'name': 'x', 'password': 'longenough1'}).status_code == 409
def test_google_off_without_client_id():
    assert TestClient(app).post('/api/auth/google', json={'credential': 'x'}).status_code == 503
def test_validation():
    c = user('d@t.io'); tid = c.get('/api/search?q=Heat').json()[0]['id']
    assert c.put('/api/diary', json={'title_id': tid, 'status': 'watched', 'rating': 9}).status_code == 400
    assert c.put('/api/diary', json={'title_id': 'zzz', 'status': 'want'}).status_code == 400
def test_recs_respect_constraints_and_not_for_me():
    c = user('e@t.io'); fl = c.get('/api/onboarding/films?genres=Sci-Fi').json()
    c.put('/api/prefs', json={'genres': ['Sci-Fi', 'Drama', 'Thriller'], 'picks': [f['id'] for f in fl[:5]], 'onboarded': True})
    r = c.post('/api/like-this-but', json={'query': 'like Interstellar, but under two hours', 'mode': 'deterministic'}).json()
    assert r['items'] and all((i['runtime'] or 999) <= 120 for i in r['items'])
    first = r['items'][0]['id']; c.post('/api/feedback/not-for-me', json={'title_id': first})
    r2 = c.post('/api/like-this-but', json={'query': 'like Interstellar, but under two hours', 'mode': 'deterministic'}).json(); assert first not in [i['id'] for i in r2['items']]
    assert not any(cl for i in r['items'] for cl in i['claims'] if cl['type'] == 'genre' and not set(cl['evidence']) <= set(i['genres']))
def test_group_needs_second_person():
    c = user('f@t.io'); assert c.post('/api/group', json={'guests': []}).status_code == 400
def test_no_invented_scores():
    t = TestClient(app).get('/api/search?q=Heat').json()[0]; assert 'imdb' not in str(t).lower() and 'rotten' not in str(t).lower()
import pytest
@pytest.mark.skipif(not os.path.exists(os.environ.get('ODDLY_LLM', '/tmp/oddly/models/qwen.gguf')), reason='local model not present')
def test_model_led_like_this_but_is_grounded_and_traced():
    c = user('g@t.io'); fl = c.get('/api/onboarding/films?genres=Sci-Fi').json()
    c.put('/api/prefs', json={'genres': ['Sci-Fi', 'Drama', 'Thriller'], 'picks': [f['id'] for f in fl[:5]], 'onboarded': True})
    r = c.post('/api/like-this-but', json={'query': 'like Interstellar, but under two hours'}).json()
    assert r['plan']['agent'] == 'model' and r['seed']['title'] == 'Interstellar'
    assert r['items'] and all((i['runtime'] or 999) <= 120 for i in r['items'])
    tr = c.get(f"/api/trace/{r['run_id']}").json(); nodes = [t['node'] for t in tr]
    assert 'react_step' in nodes and 'review' in nodes and 'retrieve' in nodes
def test_spoiler_safe_trims_unwatched_only():
    c = user('h@t.io'); t = [x for x in c.get('/api/search?q=Heat').json() if x['description'] and '.' in x['description'][:-1]][0]
    full = t['description']; assert c.put('/api/prefs', json={'spoiler_safe': True}).json()['spoiler_safe'] is True
    cut = c.get('/api/title/' + t['id']).json(); assert len(cut['description']) <= len(full) and cut['description'] and full.startswith(cut['description'])
    c.put('/api/diary', json={'title_id': t['id'], 'status': 'watched'}); assert c.get('/api/title/' + t['id']).json()['description'] == full
    assert TestClient(app).get('/api/title/' + t['id']).json()['description'] == full   # signed-out users are not trimmed
def test_delete_account_removes_everything():
    c = user('del@t.io'); t = c.get('/api/search?q=Heat').json()[0]['id']
    c.put('/api/diary', json={'title_id': t, 'status': 'want'}); uid = c.get('/api/me').json()['id']
    assert c.delete('/api/account').json() == {'deleted': True}
    from oddly.server import store
    for tb in ('users', 'prefs', 'diary', 'sessions', 'feedback', 'traces'):
        col = 'id' if tb == 'users' else 'user_id'
        assert store._q(f'select count(*) from {tb} where {col}=?', (uid,)).fetchone()[0] == 0
    assert c.get('/api/me').status_code == 401
def test_ott_lookup_cache_budget_and_no_key(monkeypatch):
    import io, json as J
    from oddly import ott
    from oddly.server import store
    monkeypatch.delenv('WATCHMODE_API_KEY', raising=False)
    assert ott.lookup(store, 'x1', 'tt0816692')['state'] == 'not_connected'
    monkeypatch.setenv('WATCHMODE_API_KEY', 'test-key'); calls = []
    def fake(url, timeout=0):
        calls.append(url); return io.BytesIO(J.dumps([{'name': 'Netflix', 'type': 'sub', 'web_url': 'https://n'}, {'name': 'Netflix', 'type': 'sub', 'web_url': 'https://n'}, {'name': 'X', 'type': 'weird'}]).encode())
    store._q('delete from ott_cache'); store._q('delete from ott_budget'); store.c.commit()
    a = ott.lookup(store, 'tt-a', 'tt0816692', fake); b = ott.lookup(store, 'tt-a', 'tt0816692', fake)
    assert a['sources'] == [{'service': 'Netflix', 'type': 'sub', 'label': 'Subscription', 'url': 'https://n'}] and a['cached'] is False and b['cached'] is True and len(calls) == 1
    assert 'regions=IN' in calls[0] and 'test-key' in calls[0]
    assert ott.lookup(store, 'tt-b', None, fake)['state'] == 'no_id'
    store._q('delete from ott_budget'); store.c.commit(); n0 = len(calls)
    monkeypatch.setattr(ott, 'DAILY_CAP', 4); assert ott.lookup(store, 'tt-c1', 'tt1', fake)['state'] == 'ok' and ott.lookup(store, 'tt-c2', 'tt1', fake)['state'] == 'ok'
    assert ott.lookup(store, 'tt-c3', 'tt1', fake)['state'] == 'budget' and len(calls) == n0 + 2     # daily cap stops further calls
    store._q('delete from ott_budget'); store.c.commit(); monkeypatch.setattr(ott, 'DAILY_CAP', 99); monkeypatch.setattr(ott, 'MONTHLY_CAP', 2)
    assert ott.lookup(store, 'tt-m1', 'tt1', fake)['state'] == 'ok' and ott.lookup(store, 'tt-m2', 'tt1', fake)['state'] == 'budget'   # monthly cap
    store._q('delete from ott_budget'); store.c.commit(); monkeypatch.setattr(ott, 'MONTHLY_CAP', 99)
    def boom(url, timeout=0): calls.append('boom'); raise OSError('down')
    k = len(calls); assert ott.lookup(store, 'tt-d', 'tt2', boom)['state'] == 'error' and ott.lookup(store, 'tt-d', 'tt2', boom)['state'] == 'error'
    assert len(calls) == k + 1   # a failed title is not retried for 10 minutes
    assert store._q('select sum(n) from ott_budget').fetchone()[0] == 2   # the failed attempt still spent its reservation
    c = user('ott@t.io'); r = c.get('/api/search?q=Heat').json()[0]['id']
    assert 'apiKey' not in str(c.get(f'/api/title/{r}/where').json())
    assert TestClient(app).get(f'/api/title/{r}/where').status_code == 401
