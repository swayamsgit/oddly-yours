"""Model-led layer for 'Like this, but...': a ReAct loop run by a local open-weights model (Qwen2.5-1.5B-Instruct Q4, Apache-2.0, llama.cpp, CPU, no network, no account).
What the MODEL decides: which tool to call next, which title the user means, which constraints to set, and (review step) whether the shelf matches the steer.
What CODE guarantees: tool arguments are schema-constrained (grammar-decoded JSON with enums), titles must resolve in the catalogue, constraints are applied as hard filters on stored data,
the loop is capped, and any invalid output falls back to the deterministic parser. Every model call is traced with real token counts and latency."""
import os, json, time, threading
BUDGET_S = float(os.environ.get('ODDLY_LLM_BUDGET_S', '40'))
MODEL = os.environ.get('ODDLY_LLM', '/tmp/oddly/models/qwen.gguf')
GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Musical']
LANGS = ['Hindi', 'English', 'Korean', 'Japanese', 'Spanish', 'French', 'Tamil', 'Telugu', 'Malayalam', 'Bengali']
STEP = {'type': 'object', 'properties': {
    'thought': {'type': 'string'},
    'tool': {'type': 'string', 'enum': ['find_title', 'set_constraints', 'finish']},
    'title': {'type': 'string'},
    'max_runtime': {'type': 'integer'}, 'min_year': {'type': 'integer'}, 'max_year': {'type': 'integer'},
    'language': {'type': 'string', 'enum': LANGS + ['any']}, 'kind': {'type': 'string', 'enum': ['movie', 'tv', 'any']},
    'genres_any': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}}, 'genres_not': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}},
    'steer': {'type': 'string', 'maxLength': 40}}, 'required': ['thought', 'tool']}
REVIEW = {'type': 'object', 'properties': {'verdict': {'type': 'string', 'enum': ['matches', 'partly', 'misses']}, 'reason': {'type': 'string', 'maxLength': 80}, 'weak': {'type': 'array', 'items': {'type': 'integer'}}}, 'required': ['verdict', 'reason', 'weak']}
def _o(tool, props, req): return {'type': 'object', 'properties': {'thought': {'type': 'string', 'maxLength': 40}, 'tool': {'type': 'string', 'enum': [tool]}, **props}, 'required': ['thought', 'tool'] + req}
TOOLS = {'find_title': _o('find_title', {'title': {'type': 'string'}}, ['title']),
         'set_constraints': _o('set_constraints', {'max_runtime': {'type': 'integer'}, 'min_year': {'type': 'integer'}, 'max_year': {'type': 'integer'}, 'language': {'type': 'string', 'enum': LANGS + ['any']}, 'kind': {'type': 'string', 'enum': ['movie', 'tv', 'any']}, 'genres_any': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}}, 'genres_not': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}}, 'steer': {'type': 'string', 'maxLength': 40}}, ['max_runtime', 'min_year', 'max_year', 'language', 'kind', 'genres_any', 'genres_not', 'steer']),
         'finish': _o('finish', {}, [])}
SYS = ("You plan a film/TV search. Tools: find_title(title) looks up the film the user wants more like; set_constraints(...) sets hard filters "
       "(max_runtime in minutes, min_year, max_year, language, kind, genres_any, genres_not) and steer = the soft change in plain words; finish() ends planning. "
       "Call find_title first when a title is named, then set_constraints with ONLY what the user said, then finish. Never invent a title or constraint. Keep thought under 12 words. One JSON object per turn.\n"
       "Examples: 'like Heat, but newer than 2010' -> min_year 2010 (newer = min_year, older = max_year). 'under two hours' -> max_runtime 120. 'in Korean' -> language Korean. Use 0 for max_runtime/min_year/max_year, 'any' for language/kind and [] for genres when the user did not say it. Call find_title with the exact title text the user typed.")
import re
SYN = {'Comedy': ('comedy', 'funny', 'funnier', 'lighter', 'laugh'), 'Horror': ('horror', 'scary', 'scarier', 'gore'), 'Romance': ('romance', 'romantic', 'love'), 'Animation': ('animation', 'animated', 'cartoon', 'anime'), 'Crime': ('crime', 'heist', 'darker'), 'Thriller': ('thriller', 'tense', 'suspense'), 'Action': ('action',), 'Sci-Fi': ('sci-fi', 'scifi', 'science fiction', 'space'), 'Documentary': ('documentary',), 'Drama': ('drama',), 'Fantasy': ('fantasy',), 'Mystery': ('mystery',), 'War': ('war',), 'Western': ('western',), 'Musical': ('musical',), 'Adventure': ('adventure',)}
def grounded(c, query):
    """Keep only constraints the user's own words support. Anything else the model produced is dropped and counted."""
    q = query.lower(); out = {}; dropped = []
    nums = {int(n) for n in re.findall(r'\b(\d{1,4})\b', q)}; words = {'one': 1, 'two': 2, 'three': 3, 'half': 30}
    for k, v in c.items():
        ok = False
        if k == 'min_year': ok = (v in nums or v - 1 in nums) and bool(re.search(r'\b(newer|after|since|from|later than|recent)\b', q))
        elif k == 'max_year': ok = (v in nums or v + 1 in nums) and bool(re.search(r'\b(older|before|until|earlier than|prior)\b', q))
        elif k == 'max_runtime': ok = bool(re.search(r'\b(under|below|less than|shorter|max|within|short)\b', q)) and (v in nums or any(w in q for w in words) or v % 30 == 0)
        elif k == 'language': ok = v.lower() in q
        elif k == 'kind': ok = bool(re.search(r'\b(tv|series|shows?)\b', q)) if v == 'tv' else bool(re.search(r'\b(movies?|films?)\b', q))
        elif k in ('genres_any', 'genres_not'):
            gs = [g for g in v if any(w in q for w in SYN.get(g, (g.lower(),)))]; ok = bool(gs); v = gs
        if ok: out[k] = v
        else: dropped.append(k)
    return out, dropped
GEMINI_MODELS = [m for m in os.environ.get('ODDLY_GEMINI_MODELS', 'gemini-3.5-flash-lite,gemini-3.1-flash-lite').split(',') if m]  # both free-tier models, 15 RPM / 500 RPD each (AI Studio rate-limit page)
GEMINI_MODEL = GEMINI_MODELS[0]
PROVIDER = os.environ.get('ODDLY_LLM_PROVIDER', 'local')  # 'gemini' = hosted build (request text goes to Google); 'local' = on-device Qwen
QUOTA = None  # set by server: callable(key, limit, ttl) -> bool, durable and shared across instances
GEM_DAY, GEM_MIN = int(os.environ.get('ODDLY_GEM_DAY', 400)), int(os.environ.get('ODDLY_GEM_MIN', 12))  # kept below Google's free 500/day and 15/min

def _gem_slot(model):
    if QUOTA is None: return True
    t = time.gmtime()
    try: return QUOTA('gem:%s:d:%s' % (model, time.strftime('%Y%m%d', t)), GEM_DAY, 90000) and QUOTA('gem:%s:m:%s' % (model, time.strftime('%Y%m%d%H%M', t)), GEM_MIN, 120)
    except Exception: return False  # counter unreachable: do not call Google

class GeminiLLM:
    """Hosted parser. Free-tier only: if Google answers 429 or any billing/quota error we stop calling for a cooldown and the deterministic parser takes over.
    There is no paid fallback. Only the typed request text is sent; the review step is disabled in the hosted build, so no shelf titles go out. No account, diary or id data."""
    URL = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent'
    def __init__(self): self.cool = {}; self.lock = threading.Lock(); self.calls = 0
    @property
    def cool_until(self): return min(self.cool.get(m, 0) for m in GEMINI_MODELS)
    @cool_until.setter
    def cool_until(self, v): self.cool = {m: v for m in GEMINI_MODELS}
    def key(self): return os.environ.get('GEMINI_API_KEY', '')
    def available(self): return bool(self.key()) and any(time.time() >= self.cool.get(m, 0) for m in GEMINI_MODELS)
    def chat(self, msgs, schema, max_tokens=160):
        import httpx
        sysm = '\n'.join(m['content'] for m in msgs if m['role'] == 'system')
        contents = [{'role': 'model' if m['role'] == 'assistant' else 'user', 'parts': [{'text': m['content']}]} for m in msgs if m['role'] != 'system']
        body = {'systemInstruction': {'parts': [{'text': sysm}]}, 'contents': contents,
                'generationConfig': {'temperature': 0, 'maxOutputTokens': max(max_tokens, 400), 'responseMimeType': 'application/json', 'responseJsonSchema': schema, 'thinkingConfig': {'thinkingLevel': 'minimal'}}}
        t = time.time()
        for model in [m for m in GEMINI_MODELS if time.time() >= self.cool.get(m, 0)]:
            if not _gem_slot(model): continue  # shared free-tier budget for this model used up
            r = httpx.post(self.URL % model, headers={'x-goog-api-key': self.key()}, json=body, timeout=12)
            if r.status_code in (429, 403) or r.status_code >= 500:
                self.cool[model] = time.time() + (60 if r.status_code == 429 else 600 if r.status_code == 403 else 30); continue  # try the other free model; never a paid one
            break
        else: raise RuntimeError('gemini free quota paused on all free models; using the built-in parser')
        if r.status_code in (429, 403) or r.status_code >= 500: raise RuntimeError(f'gemini http {r.status_code}; using the built-in parser')
        r.raise_for_status(); j = r.json(); self.calls += 1
        txt = j['candidates'][0]['content']['parts'][0]['text']
        return json.loads(txt), (j.get('usageMetadata') or {}).get('totalTokenCount', 0), (time.time() - t) * 1000
class LLM:
    def __init__(self): self.m = None; self.lock = threading.Lock()
    def available(self): return os.path.exists(MODEL)
    def chat(self, msgs, schema, max_tokens=160):
        with self.lock:
            if self.m is None:
                import atexit; from llama_cpp import Llama; self.m = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False); atexit.register(self.m.close)
            t = time.time()
            r = self.m.create_chat_completion(messages=msgs, response_format={'type': 'json_object', 'schema': schema}, max_tokens=max_tokens, temperature=0)
            return json.loads(r['choices'][0]['message']['content']), r['usage']['total_tokens'], (time.time() - t) * 1000
local_llm = LLM(); gemini = GeminiLLM()
class _Router:
    @property
    def impl(self): return gemini if PROVIDER == 'gemini' else local_llm
    def available(self): return self.impl.available()
    def chat(self, *a, **k): return self.impl.chat(*a, **k)
    def provider(self): return 'gemini' if PROVIDER == 'gemini' else 'local'
llm = _Router()
ONE = {'type': 'object', 'properties': {'title': {'type': 'string'}, 'max_runtime': {'type': 'integer'}, 'min_year': {'type': 'integer'}, 'max_year': {'type': 'integer'},
    'language': {'type': 'string', 'enum': LANGS + ['any']}, 'kind': {'type': 'string', 'enum': ['movie', 'tv', 'any']},
    'genres_any': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}}, 'genres_not': {'type': 'array', 'items': {'type': 'string', 'enum': GENRES}},
    'steer': {'type': 'string', 'maxLength': 40}}, 'required': ['title', 'steer']}
def plan_one(engine, query, log):
    """Hosted build: one model call (the free tier allows 15 requests a minute, so the 3-step loop is collapsed). The model reads the request and returns the title the user named,
    hard filters and the soft steer; the same grounding code as the ReAct path then keeps only what the user's own words support and the title must resolve in the catalogue."""
    st = {'seed': None, 'seed_idx': None, 'constraints': {}, 'unmet': [], 'steer': ''}
    msgs = [{'role': 'system', 'content': SYS.split('Examples:')[0].split('Tools:')[0] + 'Return the film the user wants more like (title, exactly as they typed it, or empty), hard filters ONLY if the user said them (newer = min_year, older = max_year, max_runtime in minutes, language, kind, genres_any, genres_not) and steer = the soft change in plain words. Use 0 for unused numbers, any for language/kind, [] for genres. Never invent a title or constraint.'}, {'role': 'user', 'content': query}]
    try: a, tok, ms = llm.chat(msgs, ONE, 200)
    except Exception as e: log('llm_error', 0, {'error': str(e)[:120]}, None); return None
    qn0 = engine._norm(query); tt = engine._norm(a.get('title', ''))
    hits = engine.find(a.get('title', ''), limit=3) if tt and tt in qn0 else []
    if not hits:
        qn = engine._norm(query); cand = [t for t in engine.by_title if len(t) > 3 and f' {t} ' in f' {qn} ']
        if cand: hits = engine.find(max(cand, key=len), limit=3)
    if hits: st['seed'] = engine.rows[hits[0]]['title']; st['seed_idx'] = hits[0]
    c = {}
    for f in ('max_runtime', 'min_year', 'max_year'):
        if isinstance(a.get(f), int) and a[f] > 0: c[f] = a[f]
    if a.get('language') in LANGS: c['language'] = a['language']
    if a.get('kind') in ('movie', 'tv'): c['kind'] = a['kind']
    if a.get('genres_any'): c['genres_any'] = a['genres_any']
    if a.get('genres_not'): c['genres_not'] = a['genres_not']
    c, dr = grounded(c, query); st['dropped'] = dr; st['constraints'] = c; st['steer'] = (a.get('steer') or '')[:120]
    log('react_step', ms, {'step': 0, 'tool': 'plan_one', 'thought': 'single hosted call', 'observation': json.dumps({'seed': st['seed'], 'constraints': c})[:200]}, tok)
    return st if (st['seed_idx'] is not None or c) else None
def plan(engine, query, log, max_steps=4):
    global STEP
    if llm.provider() == 'gemini': return plan_one(engine, query, log)
    """ReAct planning. Returns a plan dict like the deterministic parser's, or None to fall back. `log(node, ms, detail, tokens)` records every step."""
    st = {'seed': None, 'seed_idx': None, 'constraints': {}, 'unmet': [], 'steer': ''}
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': query}]; done = {'find': False, 'set': False}
    t_start = time.time()
    for k in range(max_steps):
        if time.time() - t_start > BUDGET_S: log('llm_budget', 0, {'reason': 'planning time budget hit, using what was set'}, None); return st if (st['seed_idx'] is not None or st['constraints']) else None
        allowed = [t for t in ('find_title', 'set_constraints', 'finish') if not (t == 'find_title' and (st['seed_idx'] is not None or done['find'])) and not (t == 'set_constraints' and done['set']) and not (t == 'finish' and not (done['find'] and done['set']))]
        sch = {'anyOf': [TOOLS[t] for t in allowed]}  # state machine in code: a finished step cannot be repeated
        try: a, tok, ms = llm.chat(msgs, sch, 140)
        except Exception as e: log('llm_error', 0, {'error': str(e)[:120]}, None); return None
        tool = a.get('tool'); obs = ''
        if tool == 'find_title':
            done['find'] = True
            qn0 = engine._norm(query); tt = engine._norm(a.get('title', ''))
            hits = engine.find(a.get('title', ''), limit=3) if tt and tt in qn0 else []  # the model may only name a title the user typed
            if not hits:  # grounding fallback: longest catalogue title that literally appears in the user's text
                qn = engine._norm(query); cand = [t for t in engine.by_title if len(t) > 3 and f' {t} ' in f' {qn} ']
                if cand: hits = engine.find(max(cand, key=len), limit=3)
            if hits: st['seed'] = engine.rows[hits[0]]['title']; st['seed_idx'] = hits[0]; obs = 'found: ' + '; '.join(f"{engine.rows[i]['title']} ({engine.rows[i]['year']})" for i in hits)
            else: obs = 'no title in the catalogue matches that; do not guess another'
        elif tool == 'set_constraints':
            c = {}
            for f in ('max_runtime', 'min_year', 'max_year'):
                if isinstance(a.get(f), int) and a[f] > 0: c[f] = a[f]
            if a.get('language') in LANGS: c['language'] = a['language']
            if a.get('kind') in ('movie', 'tv'): c['kind'] = a['kind']
            if a.get('genres_any'): c['genres_any'] = a['genres_any']
            if a.get('genres_not'): c['genres_not'] = a['genres_not']
            c, dr = grounded(c, query); done['set'] = True; st['dropped'] = dr
            st['constraints'] = c; st['steer'] = (a.get('steer') or '')[:120]; obs = 'constraints set: ' + json.dumps(c)
        elif tool == 'finish': log('react_step', ms, {'step': k, 'tool': 'finish', 'thought': a.get('thought', '')[:160]}, tok); return st
        log('react_step', ms, {'step': k, 'tool': tool, 'thought': a.get('thought', '')[:160], 'observation': obs[:200]}, tok)
        msgs += [{'role': 'assistant', 'content': json.dumps(a)}, {'role': 'user', 'content': 'Observation: ' + obs}]
    return st  # step cap reached: use what was set, state is kept in code
def review(engine, query, idxs, log):
    """Self-reflection: the model reads verifiable facts for the shelf and says whether it matches the request. Returns (verdict, reason, weak_indices)."""
    if llm.provider() == 'gemini': return None  # hosted: the shelf is ranked with the user's own history, so it is never sent to Google. Only the typed request goes out.
    lines = [f"{n}. {engine.rows[i]['title']} ({engine.rows[i]['year']}, {engine.rows[i]['runtime'] or '?'} min, {', '.join(engine.rows[i]['genres'][:3])})" for n, i in enumerate(idxs[:8])]
    msgs = [{'role': 'system', 'content': 'You check search results against the request using ONLY the facts listed. weak = numbers of items that clearly do not fit. Reply JSON.'},
            {'role': 'user', 'content': f'Request: {query}\nResults:\n' + '\n'.join(lines)}]
    msgs[1]['content'] = msgs[1]['content'][:1500]  # catalogue text is untrusted: bounded, and the output schema only allows a verdict, a short reason and item numbers
    try: a, tok, ms = llm.chat(msgs, REVIEW, 120)
    except Exception as e: log('llm_error', 0, {'error': str(e)[:120]}, None); return None
    weak = [w for w in a.get('weak', []) if isinstance(w, int) and 0 <= w < len(idxs[:8])]
    a['reason'] = ''.join(ch for ch in str(a.get('reason', ''))[:200] if ch.isprintable()); log('review', ms, {'verdict': a.get('verdict'), 'reason': a['reason'], 'weak': weak}, tok)
    return a.get('verdict'), a.get('reason', ''), weak

def warm():
    """Page the model weights in at server start so the first user request is not the slow one."""
    if PROVIDER != 'gemini' and llm.available():
        threading.Thread(target=lambda: llm.chat([{'role': 'user', 'content': 'hi'}], {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}, 8), daemon=True).start()
