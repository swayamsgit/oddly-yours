"""Where-to-watch (India) from Watchmode's free Developer plan: non-commercial use, attribution required, free-plan cache limit 30 days.
Looked up lazily when a signed-in user opens a title, so credits (2,500/month) are spent only on titles people look at.
No key set => 'not connected'. The key is read from WATCHMODE_API_KEY and is never stored in the repo or the data folder."""
import os, json, time, urllib.request, urllib.parse
KEY_ENV = 'WATCHMODE_API_KEY'
TTL = 29 * 86400           # terms: free plan may cache non-image data for up to 30 days
COST = 2                                                            # an IMDb-ID sources lookup costs 2 credits
MONTHLY_CAP = int(os.environ.get('ODDLY_OTT_MONTHLY_CREDITS', '2200'))   # free plan is 2,500/month; the rest is reserve for mistakes and other calls
DAILY_CAP = int(os.environ.get('ODDLY_OTT_DAILY_CREDITS', '70'))         # 70 x 31 days is 2,170, so one day cannot drain the month
RETRY_AFTER = 600                                                    # seconds before a failed title is tried again
LABEL = {'sub': 'Subscription', 'free': 'Free', 'rent': 'Rent', 'buy': 'Buy', 'tve': 'With TV provider'}
def lookup(store, title_id, imdb_id, opener=urllib.request.urlopen):
    key = os.environ.get(KEY_ENV)
    if not key: return {'state': 'not_connected'}
    if not imdb_id: return {'state': 'no_id'}
    row = store._q('select fetched, payload from ott_cache where title_id=?', (title_id,)).fetchone()
    if row and time.time() - row['fetched'] < TTL: return {'state': 'ok', 'cached': True, **json.loads(row['payload'])}
    err = store._q('select fetched from ott_err where title_id=?', (title_id,)).fetchone()
    if err and time.time() - err['fetched'] < RETRY_AFTER: return {'state': 'error'}
    day, month = time.strftime('%Y-%m-%d'), time.strftime('%Y-%m')
    d = store._q('select coalesce(sum(n),0) n from ott_budget where day=?', (day,)).fetchone()['n']
    m = store._q("select coalesce(sum(n),0) n from ott_budget where substr(day,1,7)=?", (month,)).fetchone()['n']
    if d + COST > DAILY_CAP or m + COST > MONTHLY_CAP: return {'state': 'budget'}
    # reserve the credits before the call, so a crash or retry storm cannot overspend; a failed call keeps the reservation (it may have been charged)
    store._q('insert into ott_budget(day,n) values(?,?) on conflict(day) do update set n=n+?', (day, COST, COST)); store.c.commit()
    try:
        q = urllib.parse.urlencode({'apiKey': key, 'regions': 'IN'})
        with opener(f'https://api.watchmode.com/v1/title/{urllib.parse.quote(imdb_id)}/sources/?{q}', timeout=15) as r: data = json.load(r)
    except Exception:
        store._q('insert or replace into ott_err values(?,?)', (title_id, time.time())); store.c.commit(); return {'state': 'error'}
    seen, out = set(), []
    for s in data if isinstance(data, list) else []:
        k = (s.get('name'), s.get('type'))
        if k in seen or s.get('type') not in LABEL: continue
        seen.add(k); out.append({'service': s.get('name'), 'type': s.get('type'), 'label': LABEL[s['type']], 'url': s.get('web_url')})
    payload = {'sources': out, 'region': 'IN', 'fetched': int(time.time())}
    store._q('insert or replace into ott_cache values(?,?,?)', (title_id, time.time(), json.dumps(payload))); store.c.commit()
    return {'state': 'ok', 'cached': False, **payload}
