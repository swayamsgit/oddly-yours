"""Poster fallback from MoviePosterDB's documented Embed Data API. OFF unless MOVIEPOSTERDB_API_KEY is set.
Only the free preview size (s_, 200 px) is requested (size=s). Its docs say previews are hotlinkable on any site without a key,
so the key stays on the server: the key-free image URL goes to the browser, the ?k= part is stripped, and the key is never returned.
One lookup per title when a signed-in user opens it, cached. Capped per day and per month. Not used for bulk or catalogue-wide calls."""
import os, json, time, urllib.request, urllib.parse
KEY_ENV = 'MOVIEPOSTERDB_API_KEY'
TTL = 14 * 86400
DAILY_CAP = int(os.environ.get('ODDLY_MPDB_DAILY', '300'))
MONTHLY_CAP = int(os.environ.get('ODDLY_MPDB_MONTHLY', '3000'))
RETRY_AFTER = 600
def lookup(store, title_id, imdb_id, opener=urllib.request.urlopen):
    key = os.environ.get(KEY_ENV)
    if not key: return {'state': 'not_connected'}
    if not imdb_id: return {'state': 'no_id'}
    store._q('create table if not exists mp_cache(title_id text primary key, fetched real, payload text)')
    store._q('create table if not exists mp_budget(day text primary key, n int)')
    store._q('create table if not exists mp_err(title_id text primary key, fetched real)')
    row = store._q('select fetched, payload from mp_cache where title_id=?', (title_id,)).fetchone()
    if row and time.time() - row['fetched'] < TTL: return {'state': 'ok', 'cached': True, **json.loads(row['payload'])}
    err = store._q('select fetched from mp_err where title_id=?', (title_id,)).fetchone()
    if err and time.time() - err['fetched'] < RETRY_AFTER: return {'state': 'error'}
    day, month = time.strftime('%Y-%m-%d'), time.strftime('%Y-%m')
    # durable atomic caps: the counter only increments while under the cap, so concurrent callers cannot overshoot
    if not (store.quota_take('mp:d:' + day, DAILY_CAP, 90000) and store.quota_take('mp:m:' + month, MONTHLY_CAP, 32 * 86400)): return {'state': 'budget'}
    try:
        rq = urllib.request.Request('https://movieposterdb.com/api/embed/v1/movie?' + urllib.parse.urlencode({'imdb': imdb_id, 'size': 's'}), headers={'Authorization': 'Bearer ' + key})
        with opener(rq, timeout=15) as r: data = json.load(r)
    except Exception:
        store._q('insert or replace into mp_err values(?,?)', (title_id, time.time())); store.c.commit(); return {'state': 'error'}
    pp = data.get('primary_poster') or {}
    url = (pp.get('url') or '').split('?')[0]
    if not url.startswith('https://posters.movieposterdb.com/') or '/s_' not in url:
        payload = {'url': None}
    else:
        payload = {'url': url, 'page': data.get('url'), 'fetched': int(time.time())}
    store._q('insert or replace into mp_cache values(?,?,?)', (title_id, time.time(), json.dumps(payload))); store.c.commit()
    return {'state': 'ok', 'cached': False, **payload}
