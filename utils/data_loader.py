"""
Unified data loader for the dividend-growth analysis suite.

Usage
-----
from utils.data_loader import load_ticker_data
data = load_ticker_data("JNJ")      # returns dict with keys: prices, quarterly, annual, meta
"""
from __future__ import annotations

import os
import warnings
import io
import logging
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np

# Suppress noisy yfinance encoding warning
warnings.filterwarnings(
    "ignore",
    message=r"Trying to detect encoding from a tiny portion of \(\d+\) byte\(s\)\.",
    category=UserWarning,
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")


# ── Price helpers ─────────────────────────────────────────────────────────────

def _stooq_sym(ticker: str) -> str:
    """Convert plain ticker to Stooq format, e.g. JNJ → JNJ.US"""
    t = ticker.upper()
    return t if "." in t else f"{t}.US"


def fetch_prices(ticker: str, years: int = 10) -> pd.DataFrame:
    """Download daily OHLCV from Stooq (free, no key)."""
    sym = _stooq_sym(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("Date").reset_index(drop=True)
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    return df[df["Date"] >= cutoff].reset_index(drop=True)


def _yfinance_prices(ticker: str, years: int = 10) -> pd.DataFrame:
    """Fallback: download OHLCV via yfinance when Stooq returns nothing."""
    try:
        import yfinance as yf
        start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.rename(columns={"index": "Date", "Price": "Close"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in raw.columns:
                raw[col] = float("nan")
        raw["Date"] = pd.to_datetime(raw["Date"])
        return raw[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_prices(ticker: str, years: int = 10) -> pd.DataFrame:
    """Load prices from cache; refresh if stale or missing.

    Falls back to yfinance when Stooq returns no data, so any ticker works
    without needing to run nb_01 first.
    """
    os.makedirs(PRICES_DIR, exist_ok=True)
    path = os.path.join(PRICES_DIR, f"{ticker.upper()}.US.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        if not df.empty:
            latest = pd.Timestamp(df["Date"].max())
            cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(days=1)
            if latest >= cutoff:
                return df
            fresh = fetch_prices(ticker, years=years)
            if fresh.empty:
                fresh = _yfinance_prices(ticker, years=years)
            if not fresh.empty:
                df = pd.concat([df, fresh[fresh["Date"] > latest]]).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
                df.to_csv(path, index=False)
            return df
    # No cache — try Stooq, then fall back to yfinance
    df = fetch_prices(ticker, years=years)
    if df.empty:
        df = _yfinance_prices(ticker, years=years)
    if not df.empty:
        df.to_csv(path, index=False)
    return df


# ── Fundamentals helpers ──────────────────────────────────────────────────────

def load_fundamentals(ticker: str) -> pd.DataFrame:
    """Load cached quarterly fundamentals or fetch from SEC EDGAR."""
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    path = os.path.join(FUNDAMENTALS_DIR, f"{ticker.upper()}_quarterly.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["end"])
        if not df.empty:
            return df

    from reports.reports import build_quarterly_df
    df = build_quarterly_df(ticker)
    if not df.empty:
        df.to_csv(path, index=False)
    return df


# ── Metadata via yfinance ─────────────────────────────────────────────────────

def _quiet(func, default):
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return func()
    except Exception:
        return default


def fetch_meta(ticker: str) -> dict:
    """Pull metadata: company name, sector, market cap, shares outstanding, current price."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = _quiet(lambda: t.info, {})
        return {
            "name":              info.get("longName") or info.get("shortName") or ticker,
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
            "market_cap":        info.get("marketCap"),
            "shares_outstanding":info.get("sharesOutstanding"),
            "current_price":     info.get("currentPrice") or info.get("regularMarketPrice"),
            "forward_pe":        info.get("forwardPE"),
            "beta":              info.get("beta"),
            "dividend_yield":    info.get("dividendYield"),
            "ex_dividend_date":  info.get("exDividendDate"),
            "currency":          info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker}


# ── Annual aggregation ────────────────────────────────────────────────────────

def _to_annual(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate quarterly data to annual (calendar year).
    Sums flow metrics; takes last-of-year snapshot for balance sheet items.
    """
    if quarterly.empty:
        return pd.DataFrame()

    q = quarterly.copy()
    q["year"] = pd.to_datetime(q["end"]).dt.year

    flow_cols = [c for c in ["Revenue", "NetIncomeLoss", "OperatingCashFlow", "CapExRaw",
                              "FreeCashFlow", "InterestExpense"] if c in q.columns]
    snap_cols = [c for c in ["CashAndCashEquivalents", "TotalDebt", "StockholdersEquity",
                              "NetDebt", "DebtToEquity", "SharesOutstanding"] if c in q.columns]

    agg_dict = {c: "sum" for c in flow_cols}
    agg_dict.update({c: "last" for c in snap_cols})
    # DPS & EPS: take max per year — 10-K annual totals are always the largest value in the year
    for col in ["DividendsPerShare", "EarningsPerShareBasic"]:
        if col in q.columns:
            agg_dict[col] = "max"
    if "end" in q.columns:
        agg_dict["end"] = "last"

    annual = q.groupby("year").agg(agg_dict).reset_index()

    # Re-derive payout ratios on annual basis
    if "DividendsPerShare" in annual.columns and "EarningsPerShareBasic" in annual.columns:
        annual["EarningsPayoutRatio"] = (
            annual["DividendsPerShare"] / annual["EarningsPerShareBasic"].replace(0, float("nan"))
        )
    if "DividendsPerShare" in annual.columns and "FreeCashFlow" in annual.columns and "SharesOutstanding" in annual.columns:
        annual["FCFPerShare"] = annual["FreeCashFlow"] / annual["SharesOutstanding"].replace(0, float("nan"))
        annual["FCFPayoutRatio"] = annual["DividendsPerShare"] / annual["FCFPerShare"].replace(0, float("nan"))

    return annual


# ── Dividend streak ───────────────────────────────────────────────────────────

def fetch_dividend_streak(ticker: str) -> tuple[int, str]:
    """
    Return (streak, source) where streak is consecutive years of dividend growth.

    Uses yfinance dividend history (goes back decades) so the full streak is
    captured.  Falls back to a flag so the caller can use SEC-based data.
    Source is 'yfinance' or 'sec'.

    Special one-time distributions (spin-offs, special dividends) are filtered
    out before the streak is computed: any year whose annualised total is ≥ 2×
    the rolling 3-year median is treated as a special event and skipped.
    """
    try:
        import yfinance as yf
        raw = _quiet(lambda: yf.Ticker(ticker).dividends, None)
        if raw is None or raw.empty:
            return 0, "sec"

        # Normalise timezone-aware index
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        # Aggregate individual payments → annual totals
        annual = raw.resample("YE").sum()
        annual = annual[annual > 0]

        # Exclude the current (likely partial) year
        current_year = pd.Timestamp.today().year
        annual = annual[annual.index.year < current_year]

        if len(annual) < 2:
            return 0, "yfinance"

        # Filter out special / one-time distributions.
        # A year is flagged as a special distribution when its total is ≥ 2× the
        # rolling 3-year median of surrounding years, which is a strong signal of
        # a spin-off or extraordinary payment rather than a regular dividend raise.
        rolling_med = annual.rolling(3, center=True, min_periods=1).median()
        regular = annual[annual <= rolling_med * 2]

        if len(regular) < 2:
            regular = annual  # fallback: use all data if filter is too aggressive

        vals = regular.values
        streak = 0
        for i in range(len(vals) - 1, 0, -1):
            if vals[i] > vals[i - 1]:
                streak += 1
            else:
                break
        return streak, "yfinance"
    except Exception:
        return 0, "sec"


# ── Master loader ─────────────────────────────────────────────────────────────

def load_ticker_data(ticker: str, years: int = 10) -> dict[str, pd.DataFrame | dict]:
    """
    Returns a dict with:
      - "prices"    : daily OHLCV DataFrame (Date, Open, High, Low, Close, Volume)
      - "quarterly" : quarterly fundamentals DataFrame
      - "annual"    : annual fundamentals DataFrame
      - "meta"      : dict with company name, sector, market cap, etc.
    """
    ticker = ticker.upper()
    prices    = load_prices(ticker, years=years)
    quarterly = load_fundamentals(ticker)
    annual    = _to_annual(quarterly)
    meta      = fetch_meta(ticker)
    return {
        "prices":    prices,
        "quarterly": quarterly,
        "annual":    annual,
        "meta":      meta,
    }
