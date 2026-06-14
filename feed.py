import yfinance as yf
import pandas as pd
# from pymongo import MongoClient
import os
import warnings
import logging
import io
from contextlib import redirect_stdout, redirect_stderr

# Configuration
# SYMBOLS = ['AAPL.US', 'MSFT.US', 'JNJ.US', 'FDX.US', 'PEP.US']
SYMBOLS = ['AAPL']
STORAGE_TYPE = 'csv' # Options: 'csv' or 'mongo'

# Suppress noisy encoding warning that may appear when remote providers return tiny invalid payloads
warnings.filterwarnings(
    "ignore",
    message=r"Trying to detect encoding from a tiny portion of \(\d+\) byte\(s\)\.",
    category=UserWarning,
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def get_database():
    client = MongoClient("mongodb://localhost:27017/")
    return client['financial_data']

def save_data(symbol, daily_df, quarterly_df):
    if daily_df is None or daily_df.empty:
        return

    if STORAGE_TYPE == 'mongo':
        db = get_database()
        # Store daily prices
        daily_records = daily_df.reset_index().to_dict('records')
        db[f"{symbol}_daily"].insert_many(daily_records)
        # Store quarterly financials
        quarterly_records = quarterly_df.reset_index().to_dict('records')
        db[f"{symbol}_quarterly"].insert_many(quarterly_records)
    else:
        # Save to CSV
        if not os.path.exists('data'): os.makedirs('data')
        daily_df.to_csv(f'data/{symbol}_daily.csv')
        quarterly_df.to_csv(f'data/{symbol}_quarterly.csv')


def candidate_symbols(symbol):
    candidates = [symbol]

    if symbol.endswith('.US'):
        candidates.append(symbol[:-3])

    if '.' in symbol:
        candidates.append(symbol.replace('.', '-'))

    ordered = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def cache_symbol_candidates(symbol):
    candidates = [symbol]
    if not symbol.endswith('.US'):
        candidates.append(f"{symbol}.US")
    if symbol.endswith('.US'):
        candidates.append(symbol[:-3])
    candidates.append(symbol.replace('.', '-'))

    ordered = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _quiet_yf_call(func, default):
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return func()
    except Exception:
        return default


def fetch_ticker_data(symbol):
    for candidate in candidate_symbols(symbol):
        daily_data = _quiet_yf_call(
            lambda: yf.download(
                candidate,
                period="2y",
                progress=False,
                auto_adjust=False,
                threads=False,
            ),
            pd.DataFrame(),
        )

        quarterly_data = pd.DataFrame()
        try:
            if daily_data.empty:
                ticker = yf.Ticker(candidate)
                daily_data = _quiet_yf_call(
                    lambda: ticker.history(period="2y"),
                    pd.DataFrame(),
                )
            else:
                ticker = yf.Ticker(candidate)
            quarterly_data = _quiet_yf_call(
                lambda: ticker.quarterly_balance_sheet,
                pd.DataFrame(),
            )
        except Exception:
            continue

        if not daily_data.empty:
            return candidate, daily_data, quarterly_data

    return None, pd.DataFrame(), pd.DataFrame()


def load_cached_data(symbol):
    for candidate in cache_symbol_candidates(symbol):
        daily_path = f"data/{candidate}_daily.csv"
        quarterly_path = f"data/{candidate}_quarterly.csv"

        if not os.path.exists(daily_path):
            continue

        try:
            daily_data = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        except Exception:
            continue

        if daily_data.empty:
            continue

        all_nan = daily_data.isna().all().all()
        if all_nan:
            continue

        quarterly_data = pd.DataFrame()
        if os.path.exists(quarterly_path):
            try:
                quarterly_data = pd.read_csv(quarterly_path, index_col=0)
            except Exception:
                quarterly_data = pd.DataFrame()

        return candidate, daily_data, quarterly_data

    return None, pd.DataFrame(), pd.DataFrame()

def load_stock_data(symbols):
    for symbol in symbols:
        resolved_symbol, daily_data, quarterly_data = fetch_ticker_data(symbol)

        if daily_data.empty:
            cached_symbol, daily_data, quarterly_data = load_cached_data(symbol)
            if daily_data.empty:
                print(f"Skipped {symbol}: no remote or cached price data found")
                continue

            save_data(symbol, daily_data, quarterly_data)
            print(f"Loaded {symbol} from cache (source: {cached_symbol})")
            continue

        save_data(symbol, daily_data, quarterly_data)

        if resolved_symbol != symbol:
            print(f"Successfully loaded {symbol} (fetched as {resolved_symbol})")
        else:
            print(f"Successfully loaded {symbol}")

if __name__ == "__main__":
    load_stock_data(SYMBOLS)