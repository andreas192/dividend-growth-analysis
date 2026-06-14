#!/usr/bin/env python3
"""
Batch-compare notebook metrics against Dividendology theses.

Usage
-----
    python scripts/compare_tickers.py
    python scripts/compare_tickers.py --tickers PEP UNH MO
    python scripts/compare_tickers.py --csv data/dividendology_compare.csv

Runs the same core logic as nb_03 (dividend safety) and nb_05 (DDM/DCF), with:
  - Partial calendar years excluded from annual metrics
  - yfinance FCF payout when SEC data is missing or unreliable
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Project root on path when run as scripts/compare_tickers.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.config import REQUIRED_RETURN, TERMINAL_GROWTH_RATE, cagr, fmt_pct  # noqa: E402
from utils.data_loader import (  # noqa: E402
    fetch_dividend_streak,
    fetch_meta,
    filter_complete_annual,
    load_fundamentals,
    read_fundamentals_source,
    _to_annual,
)

warnings.filterwarnings("ignore")

# ── Dividendology reference set (YouTube / Substack coverage) ─────────────────

DIVIDENDOLOGY_WATCHLIST: dict[str, dict[str, str]] = {
    "ASML": {
        "topic": "Is ASML Stock a Buy Now?",
        "category": "Tech / Semiconductors",
        "stance": "bullish",
        "summary": "EUV monopoly; AI capex beneficiary; reverse DCF suggests undervalued vs growth.",
    },
    "PEP": {
        "topic": "Buyer Beware, Warning Signs",
        "category": "Consumer Staples",
        "stance": "bearish",
        "summary": "FCF barely covers dividend; leverage and slowing growth; yield is a warning.",
    },
    "UNH": {
        "topic": "Greedy While The Market Panics",
        "category": "Healthcare",
        "stance": "bullish",
        "summary": "Healthcare selloff; super investors buying; low payout, quality dividend grower.",
    },
    "MO": {
        "topic": "BIG Dividends Now And Into The Future",
        "category": "Consumer Staples (Tobacco)",
        "stance": "bullish",
        "summary": "High yield income; legacy moat; dividend king; not a growth compounder.",
    },
    "LMT": {
        "topic": "Recession Proof Portfolio And Dividends",
        "category": "Industrials / Defense",
        "stance": "bullish",
        "summary": "Government backlog; ~40–50% FCF payout; acyclical defense cash flows.",
    },
    "UPS": {
        "topic": "Buyer Beware, Dividend Might Be Gone Next Year",
        "category": "Industrials / Logistics",
        "stance": "bearish",
        "summary": "FCF payout >100%; dividend frozen; Amazon transition and margin pressure.",
    },
    "PFE": {
        "topic": "Buyer Beware, Risk Outweighs Reward",
        "category": "Healthcare",
        "stance": "bearish",
        "summary": "Dividend exceeds FCF; patent cliff; debt from acquisitions; distressed yield.",
    },
}


@dataclass
class TickerComparison:
    ticker: str
    name: str = ""
    topic: str = ""
    category: str = ""
    dividendology_stance: str = ""
    dividendology_summary: str = ""
    price: float | None = None
    div_yield: float | None = None
    forward_pe: float | None = None
    div_streak: int = 0
    div_cagr_5y: float | None = None
    latest_dps: float | None = None
    latest_fy: int | None = None
    earnings_payout: float | None = None
    fcf_payout_sec: float | None = None
    fcf_payout_yf: float | None = None
    fcf_payout_used: float | None = None
    fcf_payout_source: str = ""
    safety_score: int | None = None
    safety_label: str = ""
    ddm_fv: float | None = None
    ddm_mos: float | None = None
    dcf_fv: float | None = None
    dcf_mos: float | None = None
    notebook_signal: str = ""
    alignment: str = ""
    data_notes: list[str] = field(default_factory=list)
    sec_available: bool = False
    data_source: str = ""


def _quiet(func, default=None):
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return func()
    except Exception:
        return default


def yfinance_fcf_payout(ticker: str) -> tuple[float | None, float | None, float | None]:
    """Return (fcf_payout, total_dividends, free_cash_flow) from yfinance."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = _quiet(lambda: t.info, {})
    cf = _quiet(lambda: t.cashflow, None)
    if cf is None or cf.empty:
        return None, None, None

    fcf = None
    for row in ("Free Cash Flow", "FreeCashFlow"):
        if row in cf.index:
            vals = cf.loc[row].dropna()
            if not vals.empty:
                fcf = float(vals.iloc[0])
                break
    if fcf is None or fcf <= 0:
        return None, None, None

    div_rate = info.get("dividendRate")
    shares = info.get("sharesOutstanding")
    if div_rate and shares:
        total_div = float(div_rate) * float(shares)
        return total_div / fcf, total_div, fcf

    payout = info.get("payoutRatio")
    if payout is not None and payout > 0:
        # Approximate FCF payout from earnings payout when cash flow statement sparse
        return float(payout), None, fcf
    return None, None, fcf


def compute_safety_score(
    earnings_payout: float | None,
    fcf_payout: float | None,
    streak: int,
    debt_to_equity: float | None,
) -> tuple[int | None, str]:
    """Mirror nb_03 dividend safety scoring."""
    scores: dict[str, int] = {}

    if earnings_payout is not None and earnings_payout > 0:
        pr = earnings_payout
        if pr < 0.40:
            scores["Earnings Payout"] = 100
        elif pr < 0.60:
            scores["Earnings Payout"] = 85
        elif pr < 0.75:
            scores["Earnings Payout"] = 65
        elif pr < 1.0:
            scores["Earnings Payout"] = 40
        else:
            scores["Earnings Payout"] = 10

    if fcf_payout is not None and fcf_payout > 0:
        v = fcf_payout
        if v < 0.50:
            scores["FCF Payout"] = 100
        elif v < 0.75:
            scores["FCF Payout"] = 75
        elif v < 1.0:
            scores["FCF Payout"] = 45
        else:
            scores["FCF Payout"] = 10

    if debt_to_equity is not None and debt_to_equity > 0:
        v = debt_to_equity
        if v < 0.5:
            scores["Debt/Equity"] = 100
        elif v < 1.0:
            scores["Debt/Equity"] = 80
        elif v < 2.0:
            scores["Debt/Equity"] = 55
        else:
            scores["Debt/Equity"] = 25

    if streak >= 25:
        scores["Growth Streak"] = 100
    elif streak >= 10:
        scores["Growth Streak"] = 75
    elif streak >= 5:
        scores["Growth Streak"] = 50
    elif streak > 0:
        scores["Growth Streak"] = 25

    if not scores:
        return None, "N/A"

    overall = int(np.mean(list(scores.values())))
    label = (
        "VERY SAFE" if overall >= 80
        else "SAFE" if overall >= 60
        else "MODERATE" if overall >= 40
        else "RISKY"
    )
    return overall, label


def notebook_signal(
    safety_score: int | None,
    safety_label: str,
    fcf_payout: float | None,
    div_yield: float | None,
) -> str:
    """Map metrics to a coarse bullish / bearish / neutral notebook view."""
    if safety_score is None:
        if div_yield and div_yield >= 0.05 and fcf_payout and fcf_payout > 1.0:
            return "bearish"
        if div_yield and div_yield < 0.02:
            return "bullish-growth"
        return "neutral"

    if safety_label in ("RISKY",) or (fcf_payout is not None and fcf_payout > 1.0):
        return "bearish"
    if safety_label in ("VERY SAFE", "SAFE") and (fcf_payout is None or fcf_payout < 0.75):
        return "bullish"
    if safety_label == "MODERATE":
        return "neutral"
    return "neutral"


def alignment_notebook_vs_dividendology(notebook: str, dividendology: str) -> str:
    nb = notebook.lower()
    dl = dividendology.lower()
    if nb == dl or (nb.startswith("bullish") and dl == "bullish") or (nb == "bearish" and dl == "bearish"):
        return "aligned"
    if nb == "neutral" or dl not in ("bullish", "bearish"):
        return "partial"
    if (nb.startswith("bullish") and dl == "bearish") or (nb == "bearish" and dl == "bullish"):
        return "conflict"
    return "partial"


def compute_ddm(latest_dps: float, div_growth: float, price: float | None) -> tuple[float | None, float | None]:
    g = max(0.01, min(div_growth, 0.20))
    if REQUIRED_RETURN <= g:
        return None, None
    d1 = latest_dps * (1 + g)
    fv = d1 / (REQUIRED_RETURN - g)
    mos = (fv - price) / price if price else None
    return fv, mos


def compute_dcf(fcf_ps: float, fcf_growth: float, price: float | None) -> tuple[float | None, float | None]:
    g = max(0.01, min(fcf_growth, 0.25))
    r, tg = REQUIRED_RETURN, TERMINAL_GROWTH_RATE
    if r <= tg:
        return None, None
    pv = 0.0
    fcf_n = fcf_ps
    for _ in range(10):
        fcf_n *= 1 + g
        pv += fcf_n / ((1 + r) ** (_ + 1))
    tv = fcf_n * (1 + tg) / (r - tg)
    fv = pv + tv / ((1 + r) ** 10)
    mos = (fv - price) / price if price else None
    return fv, mos


def analyze_ticker(ticker: str, ref: dict[str, str] | None = None) -> TickerComparison:
    ticker = ticker.upper()
    ref = ref or DIVIDENDOLOGY_WATCHLIST.get(ticker, {})
    row = TickerComparison(
        ticker=ticker,
        topic=ref.get("topic", ""),
        category=ref.get("category", ""),
        dividendology_stance=ref.get("stance", ""),
        dividendology_summary=ref.get("summary", ""),
    )

    meta = fetch_meta(ticker)
    row.name = meta.get("name") or ticker
    row.price = meta.get("current_price")
    dy = meta.get("dividend_yield")
    if dy is not None:
        dy = float(dy)
        # yfinance may return 0.041 or 4.1 for 4.1% — normalise to decimal
        row.div_yield = dy / 100 if dy > 0.20 else dy
    row.forward_pe = meta.get("forward_pe")

    yf_fcf_pay, _, _ = yfinance_fcf_payout(ticker)
    row.fcf_payout_yf = yf_fcf_pay

    streak, _ = fetch_dividend_streak(ticker)
    row.div_streak = streak

    quarterly = load_fundamentals(ticker)
    row.data_source = read_fundamentals_source(ticker)
    row.sec_available = row.data_source == "sec"
    if quarterly.empty:
        row.data_notes.append("No fundamentals (SEC or yfinance)")
        row.fcf_payout_used = yf_fcf_pay
        row.fcf_payout_source = "yfinance" if yf_fcf_pay else "none"
        row.notebook_signal = notebook_signal(None, "N/A", yf_fcf_pay, row.div_yield)
        row.alignment = alignment_notebook_vs_dividendology(row.notebook_signal, row.dividendology_stance)
        if yf_fcf_pay and yf_fcf_pay > 1.0:
            row.safety_score, row.safety_label = compute_safety_score(None, yf_fcf_pay, streak, None)
        return row

    if row.data_source == "yfinance":
        row.data_notes.append("Fundamentals from yfinance (non-SEC issuer)")

    annual = filter_complete_annual(_to_annual(quarterly))
    if annual.empty:
        row.data_notes.append("No complete annual SEC rows after filtering")
        return row

    latest = annual.iloc[-1]
    row.latest_fy = int(latest["year"]) if "year" in latest.index else None

    if "DividendsPerShare" in annual.columns:
        d = annual["DividendsPerShare"].dropna()
        d = d[d > 0]
        if not d.empty:
            row.latest_dps = float(d.iloc[-1])
            if len(d) >= 6:
                row.div_cagr_5y = cagr(float(d.iloc[-6]), float(d.iloc[-1]), 5)
            elif len(d) >= 2:
                row.div_cagr_5y = cagr(float(d.iloc[0]), float(d.iloc[-1]), len(d) - 1)

    if "EarningsPayoutRatio" in annual.columns:
        ep = annual["EarningsPayoutRatio"].dropna()
        if not ep.empty:
            row.earnings_payout = float(ep.iloc[-1])

    if "FCFPayoutRatio" in annual.columns:
        fp = annual["FCFPayoutRatio"].dropna()
        fp = fp[fp > 0]
        if not fp.empty:
            row.fcf_payout_sec = float(fp.iloc[-1])

    # Prefer yfinance FCF payout when SEC missing or materially lower (quarterly sum bug)
    if row.fcf_payout_yf is not None:
        use_yf = row.fcf_payout_sec is None
        if row.fcf_payout_sec is not None:
            sec_low_vs_yf = (
                row.fcf_payout_yf > 0.9 and row.fcf_payout_sec < 0.5
            ) or (
                row.fcf_payout_yf > row.fcf_payout_sec * 1.4
                and row.fcf_payout_yf - row.fcf_payout_sec > 0.15
            )
            if sec_low_vs_yf:
                use_yf = True
                row.data_notes.append(
                    f"SEC FCF payout ({fmt_pct(row.fcf_payout_sec)}) differs from yfinance "
                    f"({fmt_pct(row.fcf_payout_yf)}); using yfinance"
                )
        if use_yf:
            row.fcf_payout_used = row.fcf_payout_yf
            row.fcf_payout_source = "yfinance"
        else:
            row.fcf_payout_used = row.fcf_payout_sec
            row.fcf_payout_source = row.data_source if row.data_source == "yfinance" else "sec"
    elif row.fcf_payout_sec is not None:
        row.fcf_payout_used = row.fcf_payout_sec
        row.fcf_payout_source = row.data_source if row.data_source == "yfinance" else "sec"

    debt_to_equity = None
    if "DebtToEquity" in annual.columns:
        de = annual["DebtToEquity"].dropna()
        de = de[de > 0]
        if not de.empty:
            debt_to_equity = float(de.iloc[-1])

    row.safety_score, row.safety_label = compute_safety_score(
        row.earnings_payout, row.fcf_payout_used, row.div_streak, debt_to_equity
    )

    if row.latest_dps and row.div_cagr_5y is not None and row.price:
        row.ddm_fv, row.ddm_mos = compute_ddm(row.latest_dps, row.div_cagr_5y, row.price)

    fcf_ps = None
    if "FreeCashFlow" in annual.columns and "SharesOutstanding" in latest.index:
        sh = latest.get("SharesOutstanding")
        fcf = latest.get("FreeCashFlow")
        if sh and fcf and sh > 0 and fcf > 0:
            fcf_ps = float(fcf) / float(sh)

    if fcf_ps and row.div_cagr_5y is not None and row.price:
        fcf_g = row.div_cagr_5y
        if "FreeCashFlow" in annual.columns:
            fcf_data = annual[["year", "FreeCashFlow"]].dropna()
            fcf_data = fcf_data[fcf_data["FreeCashFlow"] > 0]
            if len(fcf_data) >= 6:
                fcf_g = cagr(
                    float(fcf_data.iloc[-6]["FreeCashFlow"]),
                    float(fcf_data.iloc[-1]["FreeCashFlow"]),
                    5,
                )
        row.dcf_fv, row.dcf_mos = compute_dcf(fcf_ps, fcf_g, row.price)

    row.notebook_signal = notebook_signal(
        row.safety_score, row.safety_label, row.fcf_payout_used, row.div_yield
    )
    row.alignment = alignment_notebook_vs_dividendology(
        row.notebook_signal, row.dividendology_stance
    )
    return row


def comparison_to_dataframe(rows: list[TickerComparison]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for r in rows:
        records.append({
            "Ticker": r.ticker,
            "Name": r.name,
            "Dividendology Topic": r.topic,
            "Category": r.category,
            "DL Stance": r.dividendology_stance,
            "Price": r.price,
            "Yield": r.div_yield,
            "Fwd P/E": r.forward_pe,
            "Div Streak": r.div_streak,
            "5Y Div CAGR": r.div_cagr_5y,
            "Latest DPS": r.latest_dps,
            "FY": r.latest_fy,
            "Earn Payout": r.earnings_payout,
            "FCF Payout (SEC)": r.fcf_payout_sec,
            "FCF Payout (YF)": r.fcf_payout_yf,
            "FCF Payout Used": r.fcf_payout_used,
            "FCF Source": r.fcf_payout_source,
            "Safety Score": r.safety_score,
            "Safety Label": r.safety_label,
            "DDM FV": r.ddm_fv,
            "DDM MoS": r.ddm_mos,
            "DCF FV": r.dcf_fv,
            "DCF MoS": r.dcf_mos,
            "Notebook Signal": r.notebook_signal,
            "Alignment": r.alignment,
            "Data Source": r.data_source or ("sec" if r.sec_available else "none"),
            "Notes": "; ".join(r.data_notes),
        })
    return pd.DataFrame(records)


def print_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("  Dividendology vs Notebook Comparison")
    print("=" * 72)

    for _, row in df.iterrows():
        print(f"\n── {row['Ticker']} · {row['Name']} ──")
        if row["Dividendology Topic"]:
            print(f"   Topic: {row['Dividendology Topic']} ({row['Category']})")
        print(f"   Dividendology: {row['DL Stance'].upper()}  |  Notebook: {row['Notebook Signal']}  |  {row['Alignment'].upper()}")

        parts = []
        if pd.notna(row["Price"]):
            parts.append(f"price=${row['Price']:.2f}")
        if pd.notna(row["Yield"]):
            parts.append(f"yield={row['Yield']*100:.1f}%")
        if pd.notna(row["Fwd P/E"]):
            parts.append(f"fwd P/E={row['Fwd P/E']:.1f}")
        if parts:
            print(f"   {' · '.join(parts)}")

        if pd.notna(row["FCF Payout Used"]):
            src = row["FCF Source"]
            print(f"   FCF payout: {row['FCF Payout Used']*100:.1f}% ({src})")
        if pd.notna(row["Safety Score"]):
            print(f"   Safety: {int(row['Safety Score'])} — {row['Safety Label']}")
        if pd.notna(row["5Y Div CAGR"]):
            print(f"   5Y div CAGR: {row['5Y Div CAGR']*100:.1f}% · streak: {int(row['Div Streak'])} yrs")
        if pd.notna(row["DCF MoS"]):
            mos = row["DCF MoS"] * 100
            tag = "discount" if mos > 0 else "premium"
            print(f"   DCF fair value: ${row['DCF FV']:.0f} ({abs(mos):.0f}% {tag} vs price)")
        if row["Notes"]:
            print(f"   Note: {row['Notes']}")

    aligned = (df["Alignment"] == "aligned").sum()
    partial = (df["Alignment"] == "partial").sum()
    conflict = (df["Alignment"] == "conflict").sum()
    print(f"\n{'─' * 72}")
    print(f"Summary: {aligned} aligned · {partial} partial · {conflict} conflict · {len(df)} total")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare notebook metrics to Dividendology theses.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=list(DIVIDENDOLOGY_WATCHLIST.keys()),
        help="Tickers to analyze (default: Dividendology 7-stock list)",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional path to write CSV output (e.g. data/dividendology_compare.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [t.upper() for t in args.tickers]

    rows = [analyze_ticker(t, DIVIDENDOLOGY_WATCHLIST.get(t)) for t in tickers]
    df = comparison_to_dataframe(rows)
    print_report(df)

    if args.csv:
        out = args.csv
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
