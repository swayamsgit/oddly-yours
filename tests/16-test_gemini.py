import json, httpx, pytest
from oddly import llm_agent as LA

def run(handler, monkeypatch):
    monkeypatch.setattr(httpx, 'post', lambda url, **k: handler(url, k)); monkeypatch.setenv('GEMINI_API_KEY', 'SECRETKEY123'); LA.gemini.cool_until = 0

def resp(code, body=None): return httpx.Response(code, json=body or {}, request=httpx.Request('POST', 'http://x'))

def test_parses_and_sends_only_query(monkeypatch):
    seen = {}
    def h(url, k):
        seen.update(k['json']); seen['hdr'] = k['headers']
        return resp(200, {'candidates': [{'content': {'parts': [{'text': '{"thought":"x","tool":"finish"}'}]}}], 'usageMetadata': {'totalTokenCount': 42}})
    run(h, monkeypatch)
    a, tok, ms = LA.gemini.chat([{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'like Heat'}], LA.TOOLS['finish'])
    assert a['tool'] == 'finish' and tok == 42 and 'like Heat' in json.dumps(seen['contents']) and 'SECRETKEY123' not in json.dumps({k: v for k, v in seen.items() if k != 'hdr'})

def test_429_pauses_and_falls_back(monkeypatch):
    urls = []
    def h(u, k): urls.append(u); return resp(429)
    run(h, monkeypatch)
    with pytest.raises(RuntimeError): LA.gemini.chat([{'role': 'user', 'content': 'q'}], LA.TOOLS['finish'])
    assert len(urls) == 2 and not LA.gemini.available()   # both free models tried once, then cooldown: no more calls, no paid retry
    LA.gemini.cool_until = 0

def test_single_call_plan(monkeypatch):
    from oddly.rec import Engine
    e = Engine(); n = []
    def h(url, k):
        n.append(1); return resp(200, {'candidates': [{'content': {'parts': [{'text': '{"title":"Heat","min_year":2010,"steer":"newer","max_runtime":0,"language":"any","kind":"any","genres_any":[],"genres_not":[]}'}]}}]})
    run(h, monkeypatch); monkeypatch.setattr(LA, 'PROVIDER', 'gemini')
    st = LA.plan(e, 'like Heat, but newer than 2010', lambda *a: None)
    assert len(n) == 1 and st['seed'] == 'Heat' and st['constraints'] == {'min_year': 2010}

def test_no_key_not_available(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False); assert not LA.gemini.available()

def test_review_never_sends_shelf_in_hosted_build(monkeypatch):
    monkeypatch.setattr(LA, 'PROVIDER', 'gemini')
    called = []; monkeypatch.setattr(LA.gemini, 'chat', lambda *a, **k: called.append(1))
    assert LA.review(None, 'q', [0], lambda *a: None) is None and not called
