"""
Market Dashboard Data Fetcher
Fetches EOD data from Yahoo Finance (no API key required)
Runs via GitHub Actions daily at 22:00 UTC (6am HKT next day)
Outputs: data/data.json
"""

import json
import time
import datetime
import sys
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "requests"])
    import yfinance as yf

import requests

# ── ALL TICKERS ────────────────────────────────────────────
ETF_MAIN    = ['SPY','QQQ','DIA','IWM']
SUBMARKET   = ['IVW','IVE','IJK','IJJ','IJT','IJS','MGK','VUG','VTV']
SECTOR      = ['XLK','XLV','XLF','XLE','XLY','XLI','XLB','XLU','XLRE','XLC','XLP']
SECTOR_EW   = ['RYT','RYH','RYF','RYE','RCD','RGI','RTM','RYU','EWRE','EWCO','RHS']
THEMATIC    = ['BOTZ','HACK','SOXX','ICLN','SKYY','XBI','ITA','FINX','ARKG','URA',
               'AIQ','CIBR','ROBO','ARKK','DRIV','OGIG','ACES','PAVE','HERO','CLOU']
COUNTRY     = ['EWJ','EWY','INDA','MCHI','GXC','EWH','EWU','EWQ','EWG','EWZ','EWT',
               'EWA','EWC','EWL','EWP','EWS','TUR','EWM','EPHE','THD','VNM','EWI',
               'EWN','EWD','EWK','EWO']
FUTURES     = ['ES=F','NQ=F','RTY=F','YM=F']          # Yahoo futures format
METALS      = ['GC=F','SI=F','HG=F','PL=F','PA=F']
ENERGY      = ['CL=F','NG=F']
GLOBAL_IDX  = ['^N225','^KS11','^NSEI','000001.SS','000300.SS','^HSI','^FTSE','^FCHI','^GDAXI']
YIELDS      = ['^IRX','^TNX','^TYX']  # 13wk, 10yr, 30yr treasury
DX_VIX      = ['DX-Y.NYB','^VIX']
CRYPTO_YF   = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD']

# Remap Yahoo ticker → dashboard sym
TICKER_REMAP = {
    'ES=F':'ES1!', 'NQ=F':'NQ1!', 'RTY=F':'RTY1!', 'YM=F':'YM1!',
    'GC=F':'GC1!', 'SI=F':'SI1!', 'HG=F':'HG1!', 'PL=F':'PL1!', 'PA=F':'PA1!',
    'CL=F':'CL1!', 'NG=F':'NG1!',
    '^IRX':'US3M', '^TNX':'US10Y', '^TYX':'US30Y',
    'DX-Y.NYB':'DX-Y.NYB', '^VIX':'CBOE:VIX',
    'BTC-USD':'BTC','ETH-USD':'ETH','SOL-USD':'SOL','XRP-USD':'XRP',
}

def pct(new, old):
    if old and old != 0:
        return round((new - old) / abs(old) * 100, 2)
    return 0.0

def fetch_batch(tickers, retries=3):
    """Fetch a batch of tickers from Yahoo Finance. Returns dict of sym→data."""
    results = {}
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers,
                period='1y',
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        print(f"  All retries failed for batch: {tickers[:3]}...")
        return results

    # Handle single vs multi ticker response
    if len(tickers) == 1:
        sym = tickers[0]
        df = data
        try:
            results[sym] = extract_metrics(df, sym)
        except Exception as e:
            print(f"  Error extracting {sym}: {e}")
        return results

    for sym in tickers:
        try:
            if sym in data.columns.get_level_values(0):
                df = data[sym].dropna()
            elif hasattr(data, 'columns') and sym in data:
                df = data[sym].dropna()
            else:
                continue
            results[sym] = extract_metrics(df, sym)
        except Exception as e:
            print(f"  Error extracting {sym}: {e}")
    return results

def extract_metrics(df, sym):
    """Extract price, 1d%, 1w%, hi52%, ytd%, spark[5] from a DataFrame."""
    df = df.dropna(subset=['Close'])
    if len(df) < 2:
        return None

    closes = df['Close'].values
    price  = float(closes[-1])

    # 1D change
    d1 = pct(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0

    # 1W change (5 trading days back)
    w1 = pct(closes[-1], closes[-6]) if len(closes) >= 6 else 0.0

    # 52W high
    hi52_price = float(df['High'].max()) if 'High' in df else price
    hi52_pct   = pct(price, hi52_price)

    # YTD: first trading day of this year
    this_year = datetime.datetime.now().year
    ytd_df = df[df.index.year == this_year]
    if len(ytd_df) > 0:
        ytd_start = float(ytd_df['Close'].iloc[0])
        ytd = pct(price, ytd_start)
    else:
        ytd = 0.0

    # Spark: last 5 daily 1D changes
    spark = []
    for i in range(max(1, len(closes)-5), len(closes)):
        spark.append(round(pct(closes[i], closes[i-1]), 2))
    while len(spark) < 5:
        spark.insert(0, 0.0)

    result = {
        'sym': TICKER_REMAP.get(sym, sym),
        'price': round(price, 4),
        'd1': d1,
        'w1': w1,
        'hi52': hi52_pct,
        'ytd': ytd,
        'spark': spark,
    }

    # Yields: Yahoo returns % * 100 (e.g. 4.5 = 4.5%), leave as-is
    # Crypto: add id field
    crypto_ids = {'BTC-USD':'bitcoin','ETH-USD':'ethereum','SOL-USD':'solana','XRP-USD':'ripple'}
    crypto_names = {'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','SOL-USD':'Solana','XRP-USD':'Ripple'}
    if sym in crypto_ids:
        result['id']   = crypto_ids[sym]
        result['name'] = crypto_names[sym]

    return result

def fetch_all():
    output = {
        'generated_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'futures': [],
        'dxvix': [],
        'metals': [],
        'commod': [],
        'yields': [],
        'global': [],
        'etfmain': [],
        'submarket': [],
        'sector': [],
        'sectorew': [],
        'thematic': [],
        'country': [],
        'crypto': [],
    }

    batches = [
        ('futures',    FUTURES),
        ('etfmain',    ETF_MAIN),
        ('submarket',  SUBMARKET),
        ('sector',     SECTOR),
        ('sectorew',   SECTOR_EW),
        ('thematic',   THEMATIC),
        ('country',    COUNTRY),
        ('metals',     METALS),
        ('commod',     ENERGY),
        ('global',     GLOBAL_IDX),
        ('yields',     YIELDS),
        ('dxvix',      DX_VIX),
        ('crypto',     CRYPTO_YF),
    ]

    for key, tickers in batches:
        print(f"Fetching {key} ({len(tickers)} tickers)...")
        raw = fetch_batch(tickers)

        for yf_sym in tickers:
            rec = raw.get(yf_sym)
            if rec:
                # Map yield tickers to correct sym labels
                if key == 'yields':
                    yield_map = {'^IRX':'US3M', '^TNX':'US10Y', '^TYX':'US30Y'}
                    rec['sym'] = yield_map.get(yf_sym, rec['sym'])
                output[key].append(rec)
            else:
                print(f"  ⚠ No data for {yf_sym}")

        time.sleep(1)  # polite delay between batches

    # Rename yield key: US3M → handle by sym already
    # Sort country by 1W desc (top 10 shown in dashboard)
    output['country'].sort(key=lambda x: x.get('w1', 0), reverse=True)
    output['sector'].sort(key=lambda x: x.get('w1', 0), reverse=True)
    output['sectorew'].sort(key=lambda x: x.get('w1', 0), reverse=True)
    output['thematic'].sort(key=lambda x: x.get('w1', 0), reverse=True)
    output['submarket'].sort(key=lambda x: x.get('w1', 0), reverse=True)

    return output

if __name__ == '__main__':
    print(f"=== Market Dashboard Data Fetch ===")
    print(f"Time: {datetime.datetime.utcnow()} UTC")
    print()

    data = fetch_all()

    # Write output
    out_path = Path('data/data.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    print()
    total = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"✓ Wrote {total} records to {out_path}")
    print(f"  Generated at: {data['generated_at']}")
