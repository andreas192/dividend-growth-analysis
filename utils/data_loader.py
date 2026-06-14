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

_FUNDAMENTALS_SOURCE_FILE = "{ticker}_source.txt"

# yfinance statement row names → canonical columns (first match wins)
_YF_INCOME_ROWS: dict[str, list[str]] = {
    "Revenue": ["Total Revenue", "Operating Revenue"],
    "NetIncomeLoss": [
        "Net Income Common Stockholders",
        "Net Income",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
    "EarningsPerShareBasic": ["Basic EPS", "Diluted EPS"],
}
_YF_CASHFLOW_ROWS: dict[str, list[str]] = {
    "OperatingCashFlow": ["Operating Cash Flow"],
    "CapExRaw": ["Capital Expenditure"],
    "FreeCashFlow": ["Free Cash Flow"],
}
_YF_BALANCE_ROWS: dict[str, list[str]] = {
    "CashAndCashEquivalents": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "TotalDebt": ["Total Debt"],
    "StockholdersEquity": ["Stockholders Equity", "Common Stock Equity"],
    "SharesOutstanding": ["Ordinary Shares Number", "Share Issued"],
}


def _fundamentals_source_path(ticker: str) -> str:
    return os.path.join(FUNDAMENTALS_DIR, _FUNDAMENTALS_SOURCE_FILE.format(ticker=ticker.upper()))


def read_fundamentals_source(ticker: str) -> str:
    """Return ``sec`` or ``yfinance`` for cached fundamentals, else ``unknown``."""
    path = _fundamentals_source_path(ticker)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or "unknown"
    return "unknown"


def _write_fundamentals_source(ticker: str, source: str) -> None:
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    with open(_fundamentals_source_path(ticker), "w", encoding="utf-8") as fh:
        fh.write(source)


def _pick_yf_row(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return name
    lower = {str(idx).lower(): idx for idx in df.index}
    for name in candidates:
        key = name.lower()
        if key in lower:
            return lower[key]
    return None


def _yf_cell(df: pd.DataFrame, row: str | None, col) -> float | None:
    if row is None or df is None or df.empty or col not in df.columns:
        return None
    val = df.loc[row, col]
    if pd.isna(val):
        return None
    return float(val)


def _fp_from_end(end: pd.Timestamp) -> str:
    return f"Q{(end.month - 1) // 3 + 1}"


def _annual_dps_from_history(ticker: str) -> dict[int, float]:
    """Calendar-year DPS totals from yfinance dividend payments."""
    try:
        import yfinance as yf

        raw = _quiet(lambda: yf.Ticker(ticker).dividends, None)
        if raw is None or raw.empty:
            return {}
        if hasattr(raw.index, "tz") and raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        annual = raw.resample("YE").sum()
        return {int(ts.year): float(val) for ts, val in annual.items() if val > 0}
    except Exception:
        return {}


def _fill_row_from_statements(
    row: dict,
    col,
    income: pd.DataFrame | None,
    cashflow: pd.DataFrame | None,
    balance: pd.DataFrame | None,
) -> None:
    for stmt, mapping in (
        (income, _YF_INCOME_ROWS),
        (cashflow, _YF_CASHFLOW_ROWS),
        (balance, _YF_BALANCE_ROWS),
    ):
        if stmt is None or stmt.empty or col not in stmt.columns:
            continue
        for canonical, candidates in mapping.items():
            if canonical in row:
                continue
            yf_row = _pick_yf_row(stmt, candidates)
            val = _yf_cell(stmt, yf_row, col)
            if val is not None:
                row[canonical] = val


def _derive_fundamental_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "OperatingCashFlow" in out.columns and "CapExRaw" in out.columns and "FreeCashFlow" not in out.columns:
        out["FreeCashFlow"] = out["OperatingCashFlow"] - out["CapExRaw"].abs()
    if "DividendsPerShare" in out.columns and "EarningsPerShareBasic" in out.columns:
        out["EarningsPayoutRatio"] = (
            out["DividendsPerShare"] / out["EarningsPerShareBasic"].replace(0, float("nan"))
        )
    if "CashAndCashEquivalents" in out.columns and "TotalDebt" in out.columns:
        out["NetDebt"] = out["TotalDebt"].fillna(0) - out["CashAndCashEquivalents"].fillna(0)
    if "TotalDebt" in out.columns and "StockholdersEquity" in out.columns:
        out["DebtToEquity"] = out["TotalDebt"].fillna(0) / out["StockholdersEquity"].replace(0, float("nan"))
    return out


def _fetch_yfinance_fundamentals(ticker: str, years: int = 10) -> pd.DataFrame:
    """Build a quarterly fundamentals frame from yfinance statements (non-SEC issuers)."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    q_income = _quiet(lambda: t.quarterly_financials, None)
    q_cashflow = _quiet(lambda: t.quarterly_cashflow, None)
    q_balance = _quiet(lambda: t.quarterly_balance_sheet, None)
    a_income = _quiet(lambda: t.financials, None)
    a_cashflow = _quiet(lambda: t.cashflow, None)
    a_balance = _quiet(lambda: t.balance_sheet, None)

    stmt_sets = [
        q_income,
        q_cashflow,
        q_balance,
        a_income,
        a_cashflow,
        a_balance,
    ]
    if all(s is None or s.empty for s in stmt_sets):
        return pd.DataFrame()

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    annual_dps = _annual_dps_from_history(ticker)
    records: list[dict] = []
    seen: set[tuple[pd.Timestamp, str]] = set()

    def add_record(end: pd.Timestamp, fp: str, col, income, cashflow, balance) -> None:
        key = (end.normalize(), fp)
        if key in seen or end < cutoff:
            return
        row: dict = {
            "end": end.normalize(),
            "fp": fp,
            "fy": int(end.year),
            "form": "yfinance",
            "quarter": end.to_period("Q"),
        }
        _fill_row_from_statements(row, col, income, cashflow, balance)
        if fp == "FY" and end.year in annual_dps:
            row["DividendsPerShare"] = annual_dps[end.year]
        if len(row) > 5:
            seen.add(key)
            records.append(row)

    for stmt in (q_income, q_cashflow, q_balance):
        if stmt is None or stmt.empty:
            continue
        for col in stmt.columns:
            end = pd.Timestamp(col).normalize()
            add_record(end, _fp_from_end(end), col, q_income, q_cashflow, q_balance)

    for stmt in (a_income, a_cashflow, a_balance):
        if stmt is None or stmt.empty:
            continue
        for col in stmt.columns:
            end = pd.Timestamp(col).normalize()
            add_record(end, "FY", col, a_income, a_cashflow, a_balance)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "fp"]).reset_index(drop=True)
    return _derive_fundamental_fields(df)


def load_fundamentals(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Load cached quarterly fundamentals; SEC EDGAR first, yfinance when SEC has no data."""
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    ticker = ticker.upper()
    path = os.path.join(FUNDAMENTALS_DIR, f"{ticker}_quarterly.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, parse_dates=["end"])
        if not df.empty and "fp" in df.columns and "fy" in df.columns:
            return df

    df = pd.DataFrame()
    source = "sec"
    try:
        from reports.reports import build_quarterly_df

        df = build_quarterly_df(ticker)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = _fetch_yfinance_fundamentals(ticker)
        source = "yfinance" if not df.empty else "none"
    else:
        source = "sec"

    if not df.empty:
        df.to_csv(path, index=False)
        _write_fundamentals_source(ticker, source)
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

_FLOW_COLS = [
    "Revenue", "NetIncomeLoss", "OperatingCashFlow", "CapExRaw",
    "FreeCashFlow", "InterestExpense",
]
_SNAP_COLS = [
    "CashAndCashEquivalents", "TotalDebt", "StockholdersEquity",
    "NetDebt", "DebtToEquity", "SharesOutstanding",
]
_PER_SHARE_COLS = ["DividendsPerShare", "EarningsPerShareBasic"]


def filter_complete_annual(annual: pd.DataFrame) -> pd.DataFrame:
    """Drop partial current-year rows and obvious partial-year revenue artifacts."""
    if annual.empty:
        return annual
    a = annual.copy()
    current_year = pd.Timestamp.today().year
    if "year" in a.columns:
        a = a[a["year"] < current_year]
    if "Revenue" in a.columns:
        rev = a["Revenue"].dropna()
        if not rev.empty:
            med = rev.median()
            a = a[(a["Revenue"].isna()) | (a["Revenue"] >= med * 0.25)]
    return a.reset_index(drop=True)


def _annual_year(quarterly: pd.DataFrame) -> pd.Series:
    """Calendar year from period-end date (SEC fy tags are inconsistent across concepts)."""
    return pd.to_datetime(quarterly["end"]).dt.year


def _pick_fy_value(grp: pd.DataFrame, col: str) -> float | None:
    if col not in grp.columns or "fp" not in grp.columns:
        return None
    fy_rows = grp[grp["fp"] == "FY"]
    if fy_rows.empty:
        return None
    latest_end = fy_rows["end"].max()
    at_end = fy_rows.loc[fy_rows["end"] == latest_end, col].dropna()
    if not at_end.empty:
        return float(at_end.max())
    vals = fy_rows[col].dropna()
    return float(vals.max()) if not vals.empty else None


def _pick_quarterly_sum(grp: pd.DataFrame, col: str) -> float | None:
    if col not in grp.columns:
        return None
    if "fp" not in grp.columns:
        vals = grp[col].dropna()
        return float(vals.sum()) if not vals.empty else None

    q = grp[grp["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]
    if q.empty:
        return None

    vals = q.groupby("fp")[col].last().dropna()
    if vals.empty:
        return None

    # Q4 often carries the full-year total instead of a true quarter.
    q4 = vals["Q4"] if "Q4" in vals.index else None
    q13 = vals.reindex(["Q1", "Q2", "Q3"]).dropna()
    if q4 is not None and not q13.empty and q4 >= q13.sum() * 0.9:
        return float(q4)

    return float(vals.sum())


def _pick_per_share_annual(grp: pd.DataFrame, col: str) -> float | None:
    fy_val = _pick_fy_value(grp, col)
    if fy_val is not None:
        return fy_val
    return _pick_quarterly_sum(grp, col)


def _pick_snapshot(grp: pd.DataFrame, col: str) -> float | None:
    if col not in grp.columns:
        return None
    ordered = grp.sort_values("end")
    if "fp" in ordered.columns:
        for fp in ("FY", "Q4", "Q3", "Q2", "Q1"):
            snap = ordered.loc[ordered["fp"] == fp, col].dropna()
            if not snap.empty:
                return float(snap.iloc[-1])
    vals = ordered[col].dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


def _legacy_annual_row(grp: pd.DataFrame, year: int) -> dict:
    """Fallback when quarterly rows lack fp/fy metadata (old cache)."""
    flow_cols = [c for c in _FLOW_COLS if c in grp.columns]
    snap_cols = [c for c in _SNAP_COLS if c in grp.columns]
    row: dict = {"year": year}

    for col in flow_cols:
        vals = grp[col].dropna()
        if vals.empty:
            continue
        q4 = vals.iloc[-1]
        q_rest = vals.iloc[:-1]
        row[col] = float(q4) if len(vals) > 1 and q4 >= q_rest.sum() * 0.9 else float(vals.sum())

    for col in snap_cols:
        val = _pick_snapshot(grp, col)
        if val is not None:
            row[col] = val

    for col in _PER_SHARE_COLS:
        if col not in grp.columns:
            continue
        vals = grp[col].dropna()
        if not vals.empty:
            row[col] = float(vals.max()) if col == "EarningsPerShareBasic" else float(vals.sum())

    if "end" in grp.columns:
        row["end"] = grp["end"].max()
    return row


def _to_annual(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate quarterly SEC data to fiscal-year totals.

    Flow metrics prefer FY (10-K) rows; DPS sums quarters or uses FY;
    EPS uses FY; balance-sheet items use the latest snapshot in the year.
    """
    if quarterly.empty:
        return pd.DataFrame()

    q = quarterly.copy()
    q["end"] = pd.to_datetime(q["end"])
    q["year"] = _annual_year(q)
    q = q.dropna(subset=["year"])
    q["year"] = q["year"].astype(int)

    has_fp = "fp" in q.columns and q["fp"].notna().any()
    flow_cols = [c for c in _FLOW_COLS if c in q.columns]
    snap_cols = [c for c in _SNAP_COLS if c in q.columns]

    rows: list[dict] = []
    for year, grp in q.groupby("year", sort=True):
        if has_fp:
            row: dict = {"year": year}
            for col in flow_cols:
                val = _pick_fy_value(grp, col)
                if val is None:
                    val = _pick_quarterly_sum(grp, col)
                if val is not None:
                    row[col] = val

            for col in _PER_SHARE_COLS:
                if col in grp.columns:
                    val = _pick_per_share_annual(grp, col)
                    if val is not None:
                        row[col] = val

            for col in snap_cols:
                val = _pick_snapshot(grp, col)
                if val is not None:
                    row[col] = val

            if "end" in grp.columns:
                row["end"] = grp["end"].max()
            rows.append(row)
        else:
            rows.append(_legacy_annual_row(grp, year))

    annual = pd.DataFrame(rows)

    if "OperatingCashFlow" in annual.columns and "CapExRaw" in annual.columns and "FreeCashFlow" not in annual.columns:
        annual["FreeCashFlow"] = annual["OperatingCashFlow"] - annual["CapExRaw"].abs()

    if "DividendsPerShare" in annual.columns and "EarningsPerShareBasic" in annual.columns:
        annual["EarningsPayoutRatio"] = (
            annual["DividendsPerShare"] / annual["EarningsPerShareBasic"].replace(0, float("nan"))
        )
    if (
        "DividendsPerShare" in annual.columns
        and "FreeCashFlow" in annual.columns
        and "SharesOutstanding" in annual.columns
    ):
        annual["FCFPerShare"] = annual["FreeCashFlow"] / annual["SharesOutstanding"].replace(0, float("nan"))
        annual["FCFPayoutRatio"] = annual["DividendsPerShare"] / annual["FCFPerShare"].replace(0, float("nan"))

    if "TotalDebt" in annual.columns and "StockholdersEquity" in annual.columns:
        annual["DebtToEquity"] = annual["TotalDebt"].fillna(0) / annual["StockholdersEquity"].replace(0, float("nan"))

    if "CashAndCashEquivalents" in annual.columns and "TotalDebt" in annual.columns:
        annual["NetDebt"] = annual["TotalDebt"].fillna(0) - annual["CashAndCashEquivalents"].fillna(0)

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
    annual    = filter_complete_annual(_to_annual(quarterly))
    meta      = fetch_meta(ticker)
    source    = read_fundamentals_source(ticker)
    if quarterly.empty:
        source = "none"
    meta["data_source"] = source
    return {
        "prices":    prices,
        "quarterly": quarterly,
        "annual":    annual,
        "meta":      meta,
    }
