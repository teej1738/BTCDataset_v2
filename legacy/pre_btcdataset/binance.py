import requests, pandas as pd, time, os
from datetime import datetime, timezone

URL = 'https://api.binance.com/api/v3/klines'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
INTERVALS = ['15m', '1h', '4h', '1d']
START = '2017-08-17'
OUT = os.path.join(os.path.dirname(os.path.abspath('hl.py')), 'data')

def to_ms(d):
    return int(datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ms(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')

def fetch(symbol, interval, start, end):
    all_c = []
    cur = start
    print('Fetching ' + symbol + ' ' + interval + ' from ' + from_ms(start))
    while cur < end:
        try:
            r = requests.get(URL, params={'symbol':symbol,'interval':interval,'startTime':cur,'endTime':end,'limit':1000}, timeout=15)
            batch = r.json()
        except Exception as e:
            print('Error: ' + str(e) + ' retrying...')
            time.sleep(5)
            continue
        if not batch or not isinstance(batch, list): break
        all_c.extend(batch)
        cur = batch[-1][6] + 1
        if len(all_c) % 10000 == 0:
            print('  ' + str(len(all_c)) + ' candles so far...')
        if len(batch) < 1000: break
        time.sleep(0.2)
    return all_c

def to_df(candles):
    cols = ['open_time','open','high','low','close','volume','close_time','quote_vol','trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(candles, columns=cols)
    df.drop(columns=['ignore'], inplace=True)
    for c in ['open','high','low','close','volume','quote_vol','taker_base','taker_quote']:
        df[c] = df[c].astype(float)
    df['trades'] = df['trades'].astype(int)
    df['open_time_utc'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time_utc'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df.sort_values('open_time', inplace=True)
    df.drop_duplicates(subset=['open_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

os.makedirs(OUT, exist_ok=True)
end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
start = to_ms(START)
total = len(SYMBOLS) * len(INTERVALS)
n = 0
for sym in SYMBOLS:
    for ivl in INTERVALS:
        n += 1
        print(str(n) + '/' + str(total) + ' ' + sym + ' ' + ivl)
        candles = fetch(sym, ivl, start, end)
        if not candles:
            print('No data, skipping')
            continue
        df = to_df(candles)
        today = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
        fname = sym + '_BINANCE_' + ivl + '_' + START + '_to_' + today + '.csv'
        fpath = os.path.join(OUT, fname)
        df.to_csv(fpath, index=False)
        print('Saved ' + str(len(df)) + ' candles to ' + fpath)
print('DONE')
