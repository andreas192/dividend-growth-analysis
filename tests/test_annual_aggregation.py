"""Golden tests for Phase 1 annual aggregation fixes and Phase 2 yfinance fallback."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_loader import (  # noqa: E402
    _to_annual,
    filter_complete_annual,
    load_fundamentals,
    load_ticker_data,
    read_fundamentals_source,
)


def _annual_for(ticker: str):
    quarterly = load_fundamentals(ticker, refresh=True)
    return filter_complete_annual(_to_annual(quarterly))


def _fcf_payout_yf(ticker: str) -> float | None:
    """Reference FCF payout from yfinance cash flow (dividends paid / FCF)."""
    import yfinance as yf

    cf = yf.Ticker(ticker).cashflow
    if cf is None or cf.empty:
        return None

    fcf = None
    for row in ("Free Cash Flow", "FreeCashFlow"):
        if row in cf.index:
            vals = cf.loc[row].dropna()
            if not vals.empty:
                fcf = float(vals.iloc[0])
                break
    if fcf is None or fcf <= 0:
        return None

    for row in ("Cash Dividends Paid", "Common Stock Dividend Paid"):
        if row in cf.index:
            vals = cf.loc[row].dropna()
            if not vals.empty:
                return abs(float(vals.iloc[0])) / fcf
    return None


@pytest.mark.parametrize(
    "ticker,min_div_cagr",
    [("PEP", 0.0), ("UPS", -0.05), ("LMT", 0.0)],
)
def test_dividend_cagr_sane(ticker: str, min_div_cagr: float):
    annual = _annual_for(ticker)
    dps = annual["DividendsPerShare"].dropna()
    dps = dps[dps > 0]
    assert len(dps) >= 2
    if len(dps) >= 6:
        cagr = (float(dps.iloc[-1]) / float(dps.iloc[-6])) ** (1 / 5) - 1
    else:
        cagr = (float(dps.iloc[-1]) / float(dps.iloc[0])) ** (1 / (len(dps) - 1)) - 1
    assert cagr >= min_div_cagr
    assert cagr < 0.25, f"{ticker} 5Y div CAGR looks inflated: {cagr:.1%}"


@pytest.mark.parametrize("ticker", ["PEP", "UNH", "UPS"])
def test_fcf_payout_near_yfinance(ticker: str):
    annual = _annual_for(ticker)
    sec = annual["FCFPayoutRatio"].dropna()
    assert not sec.empty, f"{ticker}: missing SEC FCF payout"
    sec_val = float(sec.iloc[-1])

    yf_val = _fcf_payout_yf(ticker)
    assert yf_val is not None, f"{ticker}: missing yfinance FCF payout reference"

    assert abs(sec_val - yf_val) <= 0.08, (
        f"{ticker}: SEC FCF payout {sec_val:.1%} vs yfinance {yf_val:.1%}"
    )


def test_unh_revenue_not_double_counted():
    annual = _annual_for("UNH")
    latest = annual.iloc[-1]
    assert float(latest["Revenue"]) < 500e9
    assert float(latest["Revenue"]) > 300e9


def test_pep_latest_dps_complete_year():
    annual = _annual_for("PEP")
    latest_dps = float(annual["DividendsPerShare"].dropna().iloc[-1])
    assert latest_dps > 4.0, f"PEP DPS looks like a partial year: {latest_dps}"


def test_asml_yfinance_fallback():
    data = load_ticker_data("ASML")
    assert data["meta"]["data_source"] == "yfinance"
    assert read_fundamentals_source("ASML") == "yfinance"
    assert not data["quarterly"].empty
    annual = data["annual"]
    assert not annual.empty
    assert float(annual["Revenue"].dropna().iloc[-1]) > 10e9
    fcf_pay = annual["FCFPayoutRatio"].dropna()
    assert not fcf_pay.empty
    assert 0.05 < float(fcf_pay.iloc[-1]) < 0.60
