"""LangGraph pipeline: plan -> retrieve (CF + embedding tools) -> rerank -> reflect (critic verifies explanation claims, retries with relaxed filters) -> respond.
Planning (title grounding, constraints) and the review step are model-led for Like-this-but when the local model is present (llm_agent.py); Tonight and Group stay deterministic. Model calls record real token counts. Every node is timed and traced to SQLite."""
import re, time, uuid, numpy as np
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from .rec import MOODS, LANG_MAP, z
from . import llm_agent as LA
class S(TypedDict, total=False):
    run_id: str; uid: int; intent: str; params: dict; plan: dict; cand: list; results: list; notes: list; attempt: int; pos: list; neg: list; genres: list; seen: set; score: Any; members: list; score_by_member: list
def parse_like_but(q):
    """'like Interstellar, but under two hours' -> seed title text + verifiable constraints. Constraints we cannot verify from data are reported, not guessed."""
    ql = q.lower().strip(); plan = {'seed': None, 'constraints': {}, 'unmet': [], 'steer': ''}
    m = re.match(r'^(?:something\s+)?like\s+(.+?)(?:,|\s+but\s+)\s*(?:but\s+)?(.*)$', ql) or re.match(r'^(?:something\s+)?like\s+(.+)$', ql)
    if m: plan['seed'] = m.group(1).strip(' ,'); rest = (m.group(2) if m.lastindex and m.lastindex >= 2 else '') or ''
    else: rest = ql
    nums = {'one': 1, 'two': 2, 'three': 3, 'an': 1}
    r = re.search(r'(?:under|below|less than|shorter than|max)\s+(\d+(?:\.\d+)?|one|two|three)\s*(hours?|hrs?|h|minutes?|mins?|m)\b', rest)
    if r:
        v = float(nums.get(r.group(1), r.group(1) if r.group(1).replace('.', '').isdigit() else 0)); plan['constraints']['max_runtime'] = int(v * 60 if r.group(2).startswith('h') else v)
    elif re.search(r'\b(shorter|quicker|not as long)\b', rest): plan['constraints']['runtime_shorter'] = True
    r = re.search(r'(?:newer than|after|since|from)\s+(\d{4})', rest)
    if r: plan['constraints']['min_year'] = int(r.group(1))
    elif re.search(r'\b(newer|more recent|modern)\b', rest): plan['constraints']['newer'] = True
    r = re.search(r'(?:older than|before)\s+(\d{4})', rest)
    if r: plan['constraints']['max_year'] = int(r.group(1))
    elif re.search(r'\b(older|classic)\b', rest): plan['constraints']['older'] = True
    for k, v in LANG_MAP.items():
        if re.search(rf'\b(in\s+)?{k}\b', rest): plan['constraints']['language'] = v
    if re.search(r'\b(tv|series|show|shows)\b', rest): plan['constraints']['kind'] = 'tv'
    elif re.search(r'\b(movie|movies|film|films)\b', rest): plan['constraints']['kind'] = 'movie'
    gmap = {'funnier': 'Comedy', 'funny': 'Comedy', 'lighter': 'Comedy', 'scarier': 'Horror', 'romantic': 'Romance', 'animated': 'Animation', 'darker': 'Crime', 'thriller': 'Thriller', 'action': 'Action', 'sci-fi': 'Sci-Fi'}
    for k, g in gmap.items():
        if k in rest: plan['constraints'].setdefault('genres_any', []).append(g)
    if re.search(r'\bno (horror|gore)\b|\bnot scary\b', rest): plan['constraints']['genres_not'] = ['Horror']
    plan['steer'] = rest
    for w in ('sadder', 'happier', 'less violent', 'no violence', 'slower', 'faster', 'darker', 'more hopeful'):
        if w in rest and w not in ('darker',): plan['unmet'].append(f'"{w}" has no reliable data behind it here, so it only nudges the text-similarity search')
    return plan
def build(engine, store):
    def timed(name):
        def deco(fn):
            def w(st):
                t = time.time(); out = fn(st); ms = (time.time() - t) * 1000
                store.trace(st['uid'], st['run_id'], name, t, ms, out.get('_d', {}) if isinstance(out, dict) else {}, None)
                out.pop('_d', None); return out
            return w
        return deco
    @timed('plan')
    def plan(st):
        it, p = st['intent'], st['params']; pl = {}
        if it == 'like_this_but':
            lg = lambda node, ms, d, tok: store.trace(st['uid'], st['run_id'], node, time.time(), ms, d, tok)
            mp = LA.plan(engine, p['query'], lg) if p.get('mode', 'model') == 'model' and LA.llm.available() else None
            if mp and (mp['seed_idx'] is not None or mp['constraints']):
                pl = {'seed': mp['seed'], 'seed_idx': mp['seed_idx'], 'constraints': mp['constraints'], 'unmet': [], 'steer': mp['steer'], 'agent': 'model'}
                if mp.get('dropped'): pl['unmet'].append('skipped filters the model suggested that that were not in your request: ' + ', '.join(mp['dropped']))
            else:
                pl = parse_like_but(p['query']); pl['agent'] = 'deterministic'; pl['seed_idx'] = engine.find(pl['seed'])[0] if pl['seed'] and engine.find(pl['seed']) else None
            if p.get('kind'): pl['constraints']['kind'] = p['kind']
        elif it == 'tonight':
            pl = {'mood': p.get('mood'), 'minutes': p.get('minutes'), 'language': p.get('language'), 'kind': p.get('kind')}
        return {'plan': pl, 'attempt': 0, 'notes': [], '_d': {'intent': it, 'plan': {k: v for k, v in pl.items() if k != 'seed_idx'}}}
    @timed('retrieve')
    def retrieve(st):
        pl, it = st['plan'], st['intent']; pos, neg, g = st['pos'], st['neg'], st['genres']; relax = st['attempt']; c = {}; extra = {}
        if it == 'like_this_but':
            si = pl.get('seed_idx'); seed = [si] if si is not None else pos
            s = engine.score_user(seed, neg, [], w_g=0)
            if si is not None and engine.has_emb[si]: s = s + 1.2 * engine.content_scores(engine.emb[si])
            if pl.get('steer'):
                v = engine.embed_query((engine.rows[si]['title'] + '. ' if si is not None else '') + pl['steer']); s = s + 0.6 * engine.content_scores(v)
            c = dict(pl['constraints']); 
            if si is not None and c.pop('runtime_shorter', False) and engine.runtime[si]: c['max_runtime'] = int(engine.runtime[si] * 0.85)
            if si is not None and c.pop('newer', False): c['min_year'] = int(engine.year[si]) + 1
            if si is not None and c.pop('older', False): c['max_year'] = int(engine.year[si]) - 1
            c.pop('runtime_shorter', None); c.pop('newer', None); c.pop('older', None)
            excl = set(st['seen']) | ({si} if si is not None else set())
        elif it == 'tonight':
            s = engine.score_user(pos, neg, g); mg = MOODS.get(pl.get('mood') or '', [])
            if mg: c['genres_any'] = mg
            if pl.get('minutes'): c['max_runtime'] = int(pl['minutes'])
            if pl.get('language'): c['language'] = pl['language']
            if pl.get('kind'): c['kind'] = pl['kind']
            excl = set(st['seen'])
        else:  # group
            ms = st['members']; sc = [engine.score_user(m['pos'], m['neg'], m['genres']) for m in ms]; extra = {'score_by_member': sc}
            s = np.mean([z(x) for x in sc], 0) + 0.7 * np.min([z(x) for x in sc], 0)
            excl = set().union(*[m['seen'] for m in ms]); c = {}
            if st['params'].get('kind'): c['kind'] = st['params']['kind']
            if st['params'].get('minutes'): c['max_runtime'] = int(st['params']['minutes'])
        if relax >= 1:  # retry: drop the least reliable constraint first
            for k in ('genres_any', 'language', 'min_year', 'max_year'):
                if k in c: c.pop(k); st['notes'].append(f'relaxed {k} to find enough matches'); break
        m = engine.mask(**{k: v for k, v in c.items() if k in ('kind', 'max_runtime', 'min_year', 'max_year', 'language', 'genres_any', 'genres_not')})
        cand = engine.top(s, m, excl, 30)
        return {**extra, 'cand': cand, 'score': s, 'notes': st['notes'], '_d': {'constraints': c, 'n_candidates': len(cand), 'attempt': relax}}
    @timed('rerank')
    def rerank(st):
        # light diversity re-rank (MMR on embeddings) so the shelf is not 12 near-duplicates
        cand = list(st['cand']); out = []; sel_v = []
        while cand and len(out) < 12:
            best, bs = None, -1e9
            for i in cand[:25]:
                red = max([float(engine.emb[i] @ v) for v in sel_v], default=0) if engine.has_emb[i] else 0
                sc = st['score'][i] - 1.4 * red
                if sc > bs: best, bs = i, sc
            cand.remove(best); out.append(best); sel_v.append(engine.emb[best])
        return {'cand': out, '_d': {'kept': len(out)}}
    @timed('reflect')
    def reflect(st):
        res = []; dropped = 0
        for i in st['cand']:
            if st['intent'] == 'group':
                cl = []; ms = st['members']
                per = [{'name': m['name'], 'fit': round(float(z(st['score_by_member'][k])[i]), 2)} for k, m in enumerate(ms)]
                res.append({'i': i, 'claims': [], 'per_member': per, 'why': 'ranked by the group: average fit, with extra weight on the least-happy member'}); continue
            cl = engine.explain(i, st['pos'], st['genres']); ok = engine.critic(i, cl, st['pos'], st['genres']); dropped += len(cl) - len(ok)
            res.append({'i': i, 'claims': ok})
        if st['intent'] == 'like_this_but' and st['plan'].get('agent') == 'model':
            rv = LA.review(engine, st['params']['query'], [r['i'] for r in res], lambda node, ms, d, tok: store.trace(st['uid'], st['run_id'], node, time.time(), ms, d, tok))
            if rv:
                v, why, weak = rv  # the review text goes to the trace (dev diagnostics) only; it misjudged verified filters in testing
                if v != 'matches' and weak and len(res) - len(weak) >= 6:
                    st['notes'].append('model-judged weak fits removed (a soft judgment, not a verified fact): ' + ', '.join(engine.rows[res[w]['i']]['title'] for w in weak)); res = [r for n, r in enumerate(res) if n not in weak]
        enough = len(res) >= 6
        return {'results': res, 'attempt': st['attempt'] + (0 if enough or st['attempt'] >= 2 else 1), '_d': {'results': len(res), 'claims_dropped_by_critic': dropped, 'retry': not enough and st['attempt'] < 2}}
    def loop(st): return 'retrieve' if len(st['results']) < 6 and st['attempt'] in (1, 2) else 'respond'
    @timed('respond')
    def respond(st): return {'_d': {'returned': len(st['results'])}}
    g = StateGraph(S)
    for n, f in (('plan', plan), ('retrieve', retrieve), ('rerank', rerank), ('reflect', reflect), ('respond', respond)): g.add_node(n, f)
    g.set_entry_point('plan'); g.add_edge('plan', 'retrieve'); g.add_edge('retrieve', 'rerank'); g.add_edge('rerank', 'reflect')
    g.add_conditional_edges('reflect', loop, {'retrieve': 'retrieve', 'respond': 'respond'}); g.add_edge('respond', END)
    return g.compile()
