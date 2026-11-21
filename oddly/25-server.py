import time, re, os, uuid, json, re, urllib.request, urllib.parse, numpy as np
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .store import Store, DATA
from .rec import Engine, MOODS, LANG_MAP
from .agent import build
HERE = os.path.dirname(__file__)
app = FastAPI(title='Oddly Yours', docs_url=None, redoc_url=None, openapi_url=None)
store = Store(); eng = Engine(); graph = build(eng, store)
from . import llm_agent as _la; _la.QUOTA = lambda k, l, t: store.quota_take(k, l, t)
from . import llm_agent as _la; _la.warm()
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')  # set via env; never hard-coded
GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Musical']
def norm_genre(g): return {'Science-Fiction': 'Sci-Fi'}.get(g, g)
for r in eng.rows: r['genres'] = [norm_genre(g) for g in r['genres']]
def me(request: Request):
    u = store.user_for(request.cookies.get('oy_session'))
    if not u: raise HTTPException(401, 'not signed in')
    return u
def me_opt(request: Request): return store.user_for(request.cookies.get('oy_session'))
def first_sentence(t):
    m = re.match(r'^(.+?[.!?])(\s|$)', t or ''); return m.group(1) if m else (t or '')
def shield(d, uid):
    """Spoiler-safe mode: for titles the user has not marked watched, show only the first sentence of the description (Wikipedia/TVmaze lead text can mention plot). This is a conservative trim, not plot-aware filtering: no reliable per-plot-point spoiler data exists in the sources."""
    if not uid or not d.get('description'): return d
    if not store.prefs(uid).get('spoiler_safe'): return d
    if d['id'] in {x['title_id'] for x in store.diary(uid) if x['status'] == 'watched'}: return d
    fs = first_sentence(d['description']); return {**d, 'description': fs, 'description_trimmed': fs != d['description']}
class Signup(BaseModel): email: str; name: str; password: str
class Login(BaseModel): email: str; password: str
class Prefs(BaseModel): genres: list[str] | None = None; picks: list[str] | None = None; services: list[str] | None = None; languages: list[str] | None = None; onboarded: bool | None = None; spoiler_safe: bool | None = None
class DiaryIn(BaseModel): title_id: str; status: str; rating: float | None = None; note: str = ''
class FB(BaseModel): title_id: str; reason: str = ''
class Tonight(BaseModel): mood: str | None = None; minutes: int | None = None; language: str | None = None; kind: str | None = None
class LikeBut(BaseModel): query: str; kind: str | None = None; mode: str = 'model'  # 'model' | 'deterministic'
class Guest(BaseModel): name: str; genres: list[str] = []; titles: list[str] = []
class Group(BaseModel): guests: list[Guest] = []; kind: str | None = None; minutes: int | None = None
def start(resp, uid): resp.set_cookie('oy_session', store.new_session(uid), httponly=True, samesite='lax', max_age=60 * 60 * 24 * 30)
@app.get('/api/config')
def config(): return {'google_client_id': GOOGLE_CLIENT_ID, 'genres': GENRES, 'moods': list(MOODS), 'languages': sorted(set(LANG_MAP.values())), 'services': ['Netflix', 'Prime Video', 'JioHotstar', 'Apple TV', 'Zee5', 'SonyLIV', 'MUBI'], 'availability_feed': bool(os.environ.get('WATCHMODE_API_KEY')), 'parser': _la.llm.provider(), 'parser_ready': _la.llm.available()}
def _ip_hash(req, scope):
    """Opaque per-day hash of the caller's IP, used only as a throttle key. The raw address is never stored or logged; the hash is salted and changes daily."""
    import hashlib
    ip = (req.headers.get('x-forwarded-for') or req.client.host or '').split(',')[0].strip()
    salt = os.environ.get('ODDLY_IP_SALT') or os.environ.get('TURSO_AUTH_TOKEN') or 'local'
    return hashlib.sha256(('%s|%s|%s|%s' % (salt, scope, time.strftime('%Y%m%d', time.gmtime()), ip)).encode()).hexdigest()[:20]
def _throttle(req, scope, per_hour, per_day_all=None):
    if req.client and req.client.host == 'testclient': return  # Starlette TestClient only
    h = time.strftime('%Y%m%d%H', time.gmtime())
    try:
        ok = store.quota_take('%s:ip:%s:%s' % (scope, _ip_hash(req, scope), h), per_hour, 3700)
        if ok and per_day_all: ok = store.quota_take('%s:all:%s' % (scope, h[:8]), per_day_all, 90000)
    except Exception: ok = False
    if not ok: raise HTTPException(429, 'Too many attempts from this network. Try again later.')
@app.post('/api/auth/signup')
def signup(b: Signup, resp: Response, req: Request):
    _throttle(req, 'signup', 5, 300)
    if len(b.password) < 8 or '@' not in b.email: raise HTTPException(400, 'Use a valid email and a password of at least 8 characters')
    try: uid = store.create_user(b.email, b.name.strip() or b.email.split('@')[0], b.password)
    except ValueError as e: raise HTTPException(409, str(e))
    start(resp, uid); return {'ok': True}
@app.post('/api/auth/login')
def login(b: Login, resp: Response, req: Request):
    _throttle(req, 'login', 30)
    uid = store.check_password(b.email, b.password)
    if not uid: raise HTTPException(401, 'Wrong email or password')
    start(resp, uid); return {'ok': True}
class GTok(BaseModel): credential: str
@app.post('/api/auth/google')
def google(b: GTok, resp: Response):
    if not GOOGLE_CLIENT_ID: raise HTTPException(503, 'Google sign-in is not configured: GOOGLE_CLIENT_ID is not set')
    try:
        with urllib.request.urlopen('https://oauth2.googleapis.com/tokeninfo?id_token=' + urllib.parse.quote(b.credential), timeout=10) as r: t = json.load(r)
    except Exception: raise HTTPException(401, 'Google could not verify that sign-in')
    if t.get('aud') != GOOGLE_CLIENT_ID or t.get('email_verified') not in ('true', True) or t.get('iss') not in ('accounts.google.com', 'https://accounts.google.com'): raise HTTPException(401, 'Google token rejected')
    uid = store.google_user(t['sub'], t['email'], t.get('name') or t['email'].split('@')[0]); start(resp, uid); return {'ok': True}
@app.delete('/api/account')
def delete_account(resp: Response, u=Depends(me)):
    """Permanently deletes the user's account, sessions, prefs, diary, feedback and traces."""
    store.delete_user(u['id']); resp.delete_cookie('oy_session'); return {'deleted': True}
@app.post('/api/auth/logout')
def logout(request: Request, resp: Response): store.end_session(request.cookies.get('oy_session')); resp.delete_cookie('oy_session'); return {'ok': True}
@app.get('/api/me')
def whoami(u=Depends(me)): return {**u, 'prefs': store.prefs(u['id']), 'counts': counts(u['id'])}
def counts(uid):
    d = store.diary(uid); return {s: sum(1 for x in d if x['status'] == s) for s in ('watched', 'want', 'dropped')}
@app.put('/api/prefs')
def put_prefs(b: Prefs, u=Depends(me)):
    kw = {k: v for k, v in b.model_dump().items() if v is not None}
    if 'genres' in kw: kw['genres'] = [g for g in kw['genres'] if g in GENRES]
    if 'picks' in kw: kw['picks'] = [t for t in kw['picks'] if t in eng.pos][:50]
    store.save_prefs(u['id'], **kw); return store.prefs(u['id'])
@app.get('/api/search')
def search(q: str = '', kind: str | None = None, limit: int = 24, u=Depends(me_opt)):
    return [shield(eng.pack(i), u and u['id']) for i in eng.find(q, kind or None, limit)]
@app.get('/api/onboarding/films')
def onboarding(genres: str = ''):
    """Popular films in the chosen genres, preferring ones with a licensed poster; used for the 5-film pick."""
    g = [x for x in genres.split(',') if x]; m = eng.mask(kind='movie', genres_any=g or None, min_year=1980)
    s = np.where(m, np.array([r['popularity'] for r in eng.rows]) + 4000 * eng.has_poster, -1); idx = np.argsort(-s)[:36]
    return [eng.pack(int(i)) for i in idx if s[i] > 0]
@app.get('/api/title/{tid}')
def title(tid: str, u=Depends(me_opt)):
    i = eng.pos.get(tid)
    if i is None: raise HTTPException(404)
    return shield(eng.pack(i), u and u['id'])
@app.get('/api/title/{tid}/where')
def where(tid: str, u=Depends(me)):
    i = eng.pos.get(tid)
    if i is None: raise HTTPException(404)
    from .ott import lookup
    return lookup(store, tid, eng.rows[i].get('imdb_id'))
@app.get('/api/title/{tid}/poster')
def mp_poster(tid: str, u=Depends(me)):
    i = eng.pos.get(tid)
    if i is None: raise HTTPException(404)
    if eng.rows[i].get('poster_url'): return {'state': 'ok', 'url': None}
    from .mpdb import lookup
    return lookup(store, tid, eng.rows[i].get('imdb_id'))
@app.get('/api/diary')
def diary(u=Depends(me)):
    out = []
    for d in store.diary(u['id']):
        i = eng.pos.get(d['title_id'])
        if i is not None: out.append({**d, 'title': shield(eng.pack(i), u['id'])})
    return out
@app.put('/api/diary')
def put_diary(b: DiaryIn, u=Depends(me)):
    if b.status not in ('watched', 'want', 'dropped') or b.title_id not in eng.pos: raise HTTPException(400, 'bad status or title')
    if b.rating is not None and not (0.5 <= b.rating <= 5): raise HTTPException(400, 'rating is 0.5 to 5')
    store.set_diary(u['id'], b.title_id, b.status, b.rating, b.note[:1000]); return {'ok': True, 'counts': counts(u['id'])}
@app.delete('/api/diary/{tid}')
def del_diary(tid: str, u=Depends(me)): store.del_diary(u['id'], tid); return {'ok': True}
@app.post('/api/feedback/not-for-me')
def nfm(b: FB, u=Depends(me)):
    if b.title_id not in eng.pos: raise HTTPException(404)
    store.add_feedback(u['id'], b.title_id, 'not_for_me', b.reason[:200]); return {'ok': True}
def ustate(uid):
    pr = store.prefs(uid); pos, neg, seen = eng.user_state(store.diary(uid), pr, store.feedback(uid)); return pr, pos, neg, seen
def run(uid, intent, params, extra=None):
    pr, pos, neg, seen = ustate(uid); st = {'run_id': uuid.uuid4().hex[:12], 'uid': uid, 'intent': intent, 'params': params, 'pos': pos, 'neg': neg, 'genres': pr['genres'], 'seen': seen}
    if extra: st.update(extra)
    out = graph.invoke(st)
    items = []
    for r in out['results']:
        d = shield(eng.pack(r['i'], {'claims': r.get('claims', []), 'per_member': r.get('per_member')}), uid)
        d['why'] = r.get('why') or ('; '.join(c['text'] for c in r['claims']) or None); items.append(d)
    return {'run_id': st['run_id'], 'items': items, 'notes': out.get('notes', []), 'plan': {k: v for k, v in out.get('plan', {}).items() if k != 'seed_idx'}, 'seed': eng.pack(out['plan']['seed_idx']) if out.get('plan', {}).get('seed_idx') is not None else None}
@app.post('/api/tonight')
def tonight(b: Tonight, u=Depends(me)):
    r = run(u['id'], 'tonight', b.model_dump())
    r['note_services'] = 'Where-to-watch (India) is in each title\'s Details, not on this list. Your services are saved but not used to filter.' if os.environ.get('WATCHMODE_API_KEY') else 'Where-to-watch is not connected in this build. Your services are saved but not used to filter.'
    return r
_RL = {}
@app.post('/api/like-this-but')
def likebut(b: LikeBut, u=Depends(me)):
    now = time.time(); hist = [t for t in _RL.get(u['id'], []) if now - t < 60]
    if len(hist) >= 6: raise HTTPException(429, 'That is a lot of requests in a minute. Try again shortly.')  # protects the shared free model quota
    _RL[u['id']] = hist + [now]
    if len(b.query) > 300: raise HTTPException(400, 'Keep the request under 300 characters.')
    if not b.query.strip(): raise HTTPException(400, 'say what you want, e.g. "like Interstellar, but under two hours"')
    return run(u['id'], 'like_this_but', b.model_dump())
@app.get('/api/home')
def home(u=Depends(me)):
    """Dashboard shelves: a hero pick, because-you-liked shelves, trending TV."""
    pr, pos, neg, seen = ustate(u['id']); s = eng.score_user(pos, neg, pr['genres'])
    hero = eng.top(s, eng.mask(kind='movie'), seen, 1)
    shelves = []
    for p in [i for i in pos if eng.has_emb[i]][-2:][::-1]:
        sc = eng.score_user([p], [], []) + eng.content_scores(eng.emb[p]); top = eng.top(sc, eng.mask(), seen | {p}, 8)
        shelves.append({'because': eng.rows[p]['title'], 'items': [shield(eng.pack(i), u['id']) for i in top]})
    tv = eng.top(s, eng.mask(kind='tv'), seen, 10)
    out = {'hero': None, 'shelves': shelves, 'tv': [shield(eng.pack(i), u['id']) for i in tv]}
    if hero:
        i = hero[0]; cl = eng.critic(i, eng.explain(i, pos, pr['genres']), pos, pr['genres']); out['hero'] = eng.pack(i, {'claims': cl, 'why': '; '.join(c['text'] for c in cl) or None})
    if out['hero']: out['hero'] = shield(out['hero'], u['id'])
    return out
@app.post('/api/group')
def group(b: Group, u=Depends(me)):
    """Common-ground mode. Other people are guests whose genres and liked titles are typed in by the signed-in user; no other account's data is read."""
    pr, pos, neg, seen = ustate(u['id']); ms = [{'name': u['name'] or 'You', 'pos': pos, 'neg': neg, 'genres': pr['genres'], 'seen': seen}]
    for g in b.guests[:5]:
        gp = [eng.pos[t] for t in g.titles if t in eng.pos]
        ms.append({'name': g.name[:30] or 'Guest', 'pos': gp, 'neg': [], 'genres': [x for x in g.genres if x in GENRES], 'seen': set(gp)})
    if len(ms) < 2: raise HTTPException(400, 'add at least one other person')
    r = run(u['id'], 'group', b.model_dump(), {'members': ms}); r['members'] = [m['name'] for m in ms]; return r
@app.get('/api/trace/{run_id}')
def trace(run_id: str, u=Depends(me)): return store.run_trace(u['id'], run_id)
@app.get('/api/evals')
def evals():
    ev = json.load(open(os.path.join(DATA, 'eval_results.json'))); ev['live_traces'] = store.trace_stats()
    ap = os.path.join(DATA, 'agent_eval.json')
    if os.path.exists(ap):
        r = json.load(open(ap)); n = len(r)
        ev['agent_eval'] = {'hosted_parser_note': 'The hosted build reads requests with Gemini (free tier). It has NOT been run through this gold set; the numbers below are for the local Qwen model and the regex parser.', 'note': 'Hand-written gold set written by the developer, 12 queries; small and not a benchmark. Local Qwen2.5-1.5B-Instruct Q4 (Apache-2.0) vs the regex parser.', 'queries': n,
            'model': {'seed_correct': sum(x['model']['seed_ok'] for x in r), 'constraints_exact': sum(x['model']['exact'] for x in r), 'ungrounded_filters_dropped': sum(len(x['model']['dropped']) for x in r), 'tokens_total': sum(x['model']['tokens'] for x in r), 'avg_ms': sum(x['model']['ms'] for x in r) // max(n, 1)},
            'regex': {'seed_correct': sum(x['regex']['seed_ok'] for x in r), 'constraints_exact': sum(x['regex']['exact'] for x in r)}}
    ev['catalogue'] = {'movies': int((eng.kind == 'movie').sum()), 'tv': int((eng.kind == 'tv').sum()), 'movies_with_licensed_poster': int(eng.has_poster[eng.kind == 'movie'].sum()), 'titles_with_text_embedding': int(eng.has_emb.sum())}
    return ev
app.mount('/', StaticFiles(directory=os.path.join(HERE, '..', 'static'), html=True), name='static')
