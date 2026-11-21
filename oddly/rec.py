"""Recommendation engine: item-kNN + SVD (MovieLens CF, movies only) fused with text-embedding similarity (movies + TV)."""
import sqlite3, json, os, re, numpy as np, scipy.sparse as sp
D = os.environ.get('ODDLY_DATA', os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
MOODS = {  # mood -> genres that fit; stated plainly in UI as a tag mapping, not a mood classifier
 'cosy': ['Comedy', 'Romance', 'Animation', 'Family', 'Children'], 'tense': ['Thriller', 'Crime', 'Mystery', 'Horror'],
 'thoughtful': ['Drama', 'Documentary', 'Science-Fiction', 'Sci-Fi', 'History'], 'laugh': ['Comedy'], 'epic': ['Adventure', 'Action', 'Fantasy', 'Sci-Fi', 'Science-Fiction', 'War'],
 'weird': ['Fantasy', 'Mystery', 'Science-Fiction', 'Sci-Fi', 'Horror']}
LANG_MAP = {'hindi': 'Hindi', 'english': 'English', 'korean': 'Korean', 'japanese': 'Japanese', 'spanish': 'Spanish', 'french': 'French', 'tamil': 'Tamil', 'telugu': 'Telugu', 'malayalam': 'Malayalam', 'bengali': 'Bengali'}
def z(a): return (a - a.mean()) / (a.std() + 1e-9)
class Engine:
    def __init__(self):
        c = sqlite3.connect(os.path.join(D, 'catalogue.sqlite')); c.row_factory = sqlite3.Row
        self.rows = [dict(r) for r in c.execute('select * from titles order by popularity desc')]
        for r in self.rows: r['genres'] = json.loads(r['genres'] or '[]')
        self.ids = [r['id'] for r in self.rows]; self.pos = {t: i for i, t in enumerate(self.ids)}; n = len(self.rows)
        self.emb = np.zeros((n, 384), np.float32); self.has_emb = np.zeros(n, bool)
        if os.path.exists(os.path.join(D, 'emb.npy')):
            E = np.load(os.path.join(D, 'emb.npy')); eid = json.load(open(os.path.join(D, 'emb_ids.json')))
            k = min(len(E), len(eid)); idx = [self.pos[t] for t in eid[:k]]; self.emb[idx] = E[:k]; self.has_emb[idx] = True
        self.emb /= (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
        self.S = sp.load_npz(os.path.join(D, 'item_knn.npz')).tocsr(); self.V = np.load(os.path.join(D, 'item_factors.npy'))
        self.ml = np.array([r['ml_idx'] if r['ml_idx'] is not None else -1 for r in self.rows]); self.is_movie = self.ml >= 0
        self.ml2row = {int(m): i for i, m in enumerate(self.ml) if m >= 0}
        self.year = np.array([r['year'] or 0 for r in self.rows]); self.runtime = np.array([r['runtime'] or 0 for r in self.rows])
        self.lang = [r['language'] for r in self.rows]; self.kind = np.array([r['kind'] for r in self.rows])
        self.has_poster = np.array([bool(r['poster_url']) for r in self.rows])
        self._qm = None; self.by_title = {}
        for i, r in enumerate(self.rows): self.by_title.setdefault(self._norm(r['title']), []).append(i)
    @staticmethod
    def _norm(s): return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
    def embed_query(self, text):
        if self._qm is None:
            from fastembed import TextEmbedding; self._qm = TextEmbedding('BAAI/bge-small-en-v1.5')
        v = np.array(list(self._qm.embed([text])))[0].astype(np.float32); return v / (np.linalg.norm(v) + 1e-9)
    def find(self, text, kind=None, limit=8):
        q = self._norm(text); out = []
        if not q: return out
        for t, ix in self.by_title.items():
            if q == t: out += [(0, i) for i in ix]
            elif t.startswith(q): out += [(1, i) for i in ix]
            elif q in t: out += [(2, i) for i in ix]
        out.sort(key=lambda x: (x[0], -self.rows[x[1]]['popularity']))
        r = [i for _, i in out if kind in (None, self.rows[i]['kind'])]; return r[:limit]
    def pack(self, i, extra=None):
        r = self.rows[i]; d = {k: r[k] for k in ('id', 'kind', 'title', 'year', 'genres', 'runtime', 'language', 'creators', 'description', 'desc_source', 'desc_url', 'community_score', 'community_source', 'poster_url', 'poster_credit', 'poster_license', 'poster_page', 'backdrop_url', 'backdrop_credit', 'backdrop_license', 'backdrop_page', 'network', 'n_ratings')}
        if extra: d.update(extra)
        return d
    # ---- scoring ----
    def cf_scores(self, pos_idx):
        """Item-kNN + SVD scores over movies from positive rows (movie rows only); returns array over all rows (0 for TV)."""
        out = np.zeros(len(self.rows)); mi = [int(self.ml[i]) for i in pos_idx if self.ml[i] >= 0]
        if not mi: return out
        n = self.S.shape[0]; x = np.zeros(n, np.float32); x[mi] = 1
        knn = self.S.T.dot(x) if False else (self.S @ x)  # S[i,j]=sim(i,j) ; score_i = sum_j in positives sim(i,j)
        u = x @ self.V / (np.sqrt(len(mi)) + 1); svd = self.V @ u
        s = z(knn) + 0.5 * z(svd); out[self.ml >= 0] = s[self.ml[self.ml >= 0]]; return out
    def content_scores(self, seed_vec):
        s = self.emb @ seed_vec; s = np.where(self.has_emb, s, np.nan); m = np.nanmean(s); sd = np.nanstd(s)
        return np.nan_to_num((s - m) / (sd + 1e-9), nan=0.0)
    def user_state(self, diary, prefs, not_for_me):
        pos, neg = [], []
        for d in diary:
            i = self.pos.get(d['title_id'])
            if i is None: continue
            if d['status'] == 'watched' and (d['rating'] is None or d['rating'] >= 3.5): pos.append(i)
            elif d['status'] == 'want': pos.append(i)
            elif d['status'] in ('dropped',) or (d['rating'] is not None and d['rating'] < 2.5): neg.append(i)
        for t in prefs['picks']:
            i = self.pos.get(t)
            if i is not None and i not in pos: pos.append(i)
        neg += [self.pos[t] for t in not_for_me if t in self.pos]
        seen = {self.pos[d['title_id']] for d in diary if d['title_id'] in self.pos} | {self.pos[t] for t in prefs['picks'] if t in self.pos}
        seen |= {self.pos[t] for t in not_for_me if t in self.pos}
        return pos, neg, seen
    def score_user(self, pos, neg, genres, w_cf=1.0, w_ct=1.0, w_g=0.35):
        n = len(self.rows); s = np.zeros(n)
        if pos:
            s += w_cf * self.cf_scores(pos)
            ev = [self.emb[i] for i in pos if self.has_emb[i]]
            if ev: c = np.mean(ev, 0); c /= np.linalg.norm(c) + 1e-9; s += w_ct * self.content_scores(c)
        else: s += 0.4 * z(np.log1p(np.array([r['popularity'] for r in self.rows])))
        if genres:
            g = np.array([len(set(genres) & set(r['genres'])) / max(1, len(genres)) for r in self.rows]); s += w_g * 3 * g
        if neg:
            ev = [self.emb[i] for i in neg if self.has_emb[i]]
            if ev: c = np.mean(ev, 0); c /= np.linalg.norm(c) + 1e-9; s -= 0.6 * self.content_scores(c)
        s += 0.1 * z(np.log1p(np.array([r['popularity'] for r in self.rows])))
        return s
    def mask(self, kind=None, max_runtime=None, min_year=None, max_year=None, language=None, genres_any=None, genres_not=None, require_poster=False):
        m = np.ones(len(self.rows), bool)
        if kind: m &= self.kind == kind
        if max_runtime: m &= (self.runtime > 0) & (self.runtime <= max_runtime)
        if min_year: m &= self.year >= min_year
        if max_year: m &= (self.year > 0) & (self.year <= max_year)
        if language: m &= np.array([l == language for l in self.lang])
        if genres_any: m &= np.array([bool(set(genres_any) & set(r['genres'])) for r in self.rows])
        if genres_not: m &= np.array([not (set(genres_not) & set(r['genres'])) for r in self.rows])
        if require_poster: m &= self.has_poster | (self.kind == 'tv')
        return m
    def top(self, score, mask, exclude, k=12):
        s = np.where(mask, score, -1e9); s[list(exclude)] = -1e9 if exclude else s[list(exclude)]
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]; idx = idx[np.argsort(-s[idx])]; return [int(i) for i in idx if s[i] > -1e8]
    # ---- grounded explanations ----
    def explain(self, i, pos, genres):
        """Claims are generated only from verifiable facts: shared genres and nearest liked titles by embedding/CF neighbours."""
        r = self.rows[i]; claims = []
        sh = [g for g in r['genres'] if g in genres]
        if sh: claims.append({'type': 'genre', 'text': f"matches your picked genres: {', '.join(sh)}", 'evidence': sh})
        if pos and self.has_emb[i]:
            cand = [(float(self.emb[i] @ self.emb[p]), p) for p in pos if self.has_emb[p] and p != i]
            if cand:
                sim, p = max(cand); 
                if sim > 0.55: claims.append({'type': 'similar', 'text': f"reads close to {self.rows[p]['title']} ({self.rows[p]['year']}), which you liked", 'evidence': [self.rows[p]['id']], 'sim': round(sim, 2)})
        if self.ml[i] >= 0:
            mi = [int(self.ml[p]) for p in pos if self.ml[p] >= 0]
            if mi:
                col = self.S[int(self.ml[i])]; best = None
                for m in mi:
                    v = col[0, m]
                    if v > (best[0] if best else 0.12): best = (float(v), m)
                if best: claims.append({'type': 'cf', 'text': f"people who rated {self.rows[self.ml2row[best[1]]]['title']} highly also rated this highly (MovieLens, cosine {best[0]:.2f})", 'evidence': [self.rows[self.ml2row[best[1]]]['id']]})
        return claims
    def critic(self, i, claims, pos, genres):
        """Self-reflection pass: re-verify each claim against data; drop any that fails."""
        ok = []
        for c in claims:
            if c['type'] == 'genre' and all(g in self.rows[i]['genres'] and g in genres for g in c['evidence']): ok.append(c)
            elif c['type'] == 'similar' and c['evidence'][0] in [self.ids[p] for p in pos]: ok.append(c)
            elif c['type'] == 'cf' and c['evidence'][0] in [self.ids[p] for p in pos]: ok.append(c)
        return ok
