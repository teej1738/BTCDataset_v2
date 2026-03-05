import requests, pandas as pd, time, os
from datetime import datetime, timezone

URL = 'https://api.hyperliquid.xyz/info'
COINS = ['BTC', 'ETH', 'SOL']
INTERVALS = ['15m', '1h', '4h', '1d']
START = '2023-10-01'
OUT = os.path.join(os.path.dirname(os.path.abspath('hl.py')), 'data')

def to_ms(d):
    return int(datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ms(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')

def ivl_ms(i):
    m = {'1m':60000,'5m':300000,'15m':900000,'30m':1800000,'1h':3600000,'2h':7200000,'4h':14400000,'12h':43200000,'1d':86400000}
    return m.get(i, 3600000)

def fetch(coin, interval, start, end):
    all_c = []
    cur = start
    win = ivl_ms(interval) * 5000
    print('Fetching ' + coin + ' ' + interval + ' from ' + from_ms(start))
    while cur < end:
        try:
            r = requests.post(URL, json={'type':'candleSnapshot','req':{'coin':coin,'interval':interval,'startTime':cur,'endTime':min(cur+win,end)}}, timeout=15)
            batch = r.json()
        except Exception as e:
            print('Error: ' + str(e) + ' retrying...')
            time.sleep(5)
            continue
        if not batch: break
        all_c.extend(batch)
        cur = batch[-1]['t'] + ivl_ms(interval)
        if len(batch) < 10: break
        time.sleep(0.3)
    return all_c

def to_df(candles):
    rows = [{'open_time':c['t'],'open':float(c['o']),'high':float(c['h']),'low':float(c['l']),'close':float(c['c']),'volume':float(c['v']),'close_time':c['T'],'trades':int(c['n'])} for c in candles]
    df = pd.DataFrame(rows)
    if df.empty: return df
    df['open_time_utc'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time_utc'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df.sort_values('open_time', inplace=True)
    df.drop_duplicates(subset=['open_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

os.makedirs(OUT, exist_ok=True)
end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
start = to_ms(START)
total = len(COINS) * len(INTERVALS)
n = 0
for coin in COINS:
    for ivl in INTERVALS:
        n += 1
        print(str(n) + '/' + str(total) + ' ' + coin + ' ' + ivl)
        candles = fetch(coin, ivl, start, end)
        if not candles:
            print('No data, skipping')
            continue
        df = to_df(candles)
        today = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
        fname = coin + '_HL_' + ivl + '_' + START + '_to_' + today + '.csv'
        fpath = os.path.join(OUT, fname)
        df.to_csv(fpath, index=False)
        print('Saved ' + str(len(df)) + ' candles to ' + fpath)
print('DONE')
