#!/usr/bin/env python3
"""Load CSV files from /data into PostgreSQL tables."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

FUNDAMENTALS_COLUMN_MAP = {
    "end": "end_date",
    "Revenue": "revenue",
    "NetIncomeLoss": "net_income_loss",
    "EarningsPerShareBasic": "earnings_per_share_basic",
    "OperatingCashFlow": "operating_cash_flow",
    "CapExRaw": "capex_raw",
    "DividendsPerShare": "dividends_per_share",
    "CashAndCashEquivalents": "cash_and_cash_equivalents",
    "TotalDebt": "total_debt",
    "StockholdersEquity": "stockholders_equity",
    "InterestExpense": "interest_expense",
    "SharesOutstanding": "shares_outstanding",
    "FreeCashFlow": "free_cash_flow",
    "EarningsPayoutRatio": "earnings_payout_ratio",
    "NetDebt": "net_debt",
    "DebtToEquity": "debt_to_equity",
    "quarter": "quarter",
    "fy": "fy",
    "fp": "fp",
    "form": "form",
}

COMPARE_COLUMN_MAP = {
    "Ticker": "ticker",
    "Name": "name",
    "Dividendology Topic": "dividendology_topic",
    "Category": "category",
    "DL Stance": "dl_stance",
    "Price": "price",
    "Yield": "yield",
    "Fwd P/E": "fwd_pe",
    "Div Streak": "div_streak",
    "5Y Div CAGR": "div_cagr_5y",
    "Latest DPS": "latest_dps",
    "FY": "fy",
    "Earn Payout": "earn_payout",
    "FCF Payout (SEC)": "fcf_payout_sec",
    "FCF Payout (YF)": "fcf_payout_yf",
    "FCF Payout Used": "fcf_payout_used",
    "FCF Source": "fcf_source",
    "Safety Score": "safety_score",
    "Safety Label": "safety_label",
    "DDM FV": "ddm_fv",
    "DDM MoS": "ddm_mos",
    "DCF FV": "dcf_fv",
    "DCF MoS": "dcf_mos",
    "Notebook Signal": "notebook_signal",
    "Alignment": "alignment",
    "SEC Data": "sec_data",
    "Notes": "notes",
}


def connect():
    kwargs = {
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }
    if host := os.environ.get("PGHOST"):
        kwargs["host"] = host
        kwargs["port"] = os.environ.get("PGPORT", "5432")
    return psycopg2.connect(**kwargs)


def _ticker_from_price_path(path: Path) -> str:
    return path.stem.replace(".US", "").upper()


def _ticker_from_fundamentals_path(path: Path) -> str:
    return path.name.replace("_quarterly.csv", "").upper()


def load_prices(cur) -> int:
    rows: list[tuple] = []
    for path in sorted(DATA_DIR.glob("prices/*.csv")):
        ticker = _ticker_from_price_path(path)
        df = pd.read_csv(path, parse_dates=["Date"])
        for _, row in df.iterrows():
            rows.append(
                (
                    ticker,
                    row["Date"].date(),
                    row.get("Open"),
                    row.get("High"),
                    row.get("Low"),
                    row.get("Close"),
                    row.get("Volume"),
                )
            )
    if not rows:
        return 0
    execute_values(
        cur,
        """
        INSERT INTO prices (ticker, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, date) DO NOTHING
        """,
        rows,
        page_size=1000,
    )
    return len(rows)


def load_fundamentals(cur) -> int:
    rows: list[tuple] = []
    db_cols = ["ticker", *FUNDAMENTALS_COLUMN_MAP.values()]

    for path in sorted(DATA_DIR.glob("fundamentals/*_quarterly.csv")):
        ticker = _ticker_from_fundamentals_path(path)
        df = pd.read_csv(path, parse_dates=["end"])
        df = df.rename(columns=FUNDAMENTALS_COLUMN_MAP)
        df["ticker"] = ticker
        if "fp" not in df.columns:
            df["fp"] = None
        df["fp"] = df["fp"].where(df["fp"].notna(), None)
        if "quarter" not in df.columns:
            df["quarter"] = df["end_date"].astype(str)
        df["quarter"] = df["quarter"].fillna(df["end_date"].astype(str))
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.date

        for _, row in df.iterrows():
            rows.append(
                tuple(
                    None if pd.isna(row.get(col)) else row.get(col)
                    for col in db_cols
                )
            )

    if not rows:
        return 0
    cols_sql = ", ".join(db_cols)
    execute_values(
        cur,
        f"""
        INSERT INTO fundamentals ({cols_sql})
        VALUES %s
        ON CONFLICT (ticker, end_date, quarter) DO NOTHING
        """,
        rows,
        page_size=500,
    )
    return len(rows)


def load_fundamentals_sources(cur) -> int:
    rows: list[tuple] = []
    for path in sorted(DATA_DIR.glob("fundamentals/*_source.txt")):
        ticker = path.name.replace("_source.txt", "").upper()
        source = path.read_text(encoding="utf-8").strip() or "unknown"
        rows.append((ticker, source))
    if not rows:
        return 0
    execute_values(
        cur,
        """
        INSERT INTO fundamentals_sources (ticker, source)
        VALUES %s
        ON CONFLICT (ticker) DO UPDATE SET source = EXCLUDED.source
        """,
        rows,
    )
    return len(rows)


def load_dividendology_compare(cur) -> int:
    path = DATA_DIR / "dividendology_compare.csv"
    if not path.exists():
        return 0

    df = pd.read_csv(path)
    df = df.rename(columns=COMPARE_COLUMN_MAP)
    if "sec_data" in df.columns:
        df["sec_data"] = df["sec_data"].map(
            lambda v: bool(v) if pd.notna(v) else None
        )

    db_cols = list(COMPARE_COLUMN_MAP.values())
    rows = [
        tuple(None if pd.isna(row.get(col)) else row.get(col) for col in db_cols)
        for _, row in df.iterrows()
    ]
    if not rows:
        return 0

    cols_sql = ", ".join(db_cols)
    execute_values(
        cur,
        f"""
        INSERT INTO dividendology_compare ({cols_sql})
        VALUES %s
        ON CONFLICT (ticker) DO UPDATE SET
            name = EXCLUDED.name,
            dividendology_topic = EXCLUDED.dividendology_topic,
            category = EXCLUDED.category,
            dl_stance = EXCLUDED.dl_stance,
            price = EXCLUDED.price,
            yield = EXCLUDED.yield,
            fwd_pe = EXCLUDED.fwd_pe,
            div_streak = EXCLUDED.div_streak,
            div_cagr_5y = EXCLUDED.div_cagr_5y,
            latest_dps = EXCLUDED.latest_dps,
            fy = EXCLUDED.fy,
            earn_payout = EXCLUDED.earn_payout,
            fcf_payout_sec = EXCLUDED.fcf_payout_sec,
            fcf_payout_yf = EXCLUDED.fcf_payout_yf,
            fcf_payout_used = EXCLUDED.fcf_payout_used,
            fcf_source = EXCLUDED.fcf_source,
            safety_score = EXCLUDED.safety_score,
            safety_label = EXCLUDED.safety_label,
            ddm_fv = EXCLUDED.ddm_fv,
            ddm_mos = EXCLUDED.ddm_mos,
            dcf_fv = EXCLUDED.dcf_fv,
            dcf_mos = EXCLUDED.dcf_mos,
            notebook_signal = EXCLUDED.notebook_signal,
            alignment = EXCLUDED.alignment,
            sec_data = EXCLUDED.sec_data,
            notes = EXCLUDED.notes
        """,
        rows,
    )
    return len(rows)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    with connect() as conn:
        with conn.cursor() as cur:
            counts = {
                "prices": load_prices(cur),
                "fundamentals": load_fundamentals(cur),
                "fundamentals_sources": load_fundamentals_sources(cur),
                "dividendology_compare": load_dividendology_compare(cur),
            }
        conn.commit()

    for table, count in counts.items():
        print(f"Loaded {count} rows into {table}")


if __name__ == "__main__":
    main()
