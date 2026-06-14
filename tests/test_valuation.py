"""Tests for Phase 3 valuation and safety helpers."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.valuation import (  # noqa: E402
    compute_dcf,
    compute_reverse_dcf,
    compute_safety_score,
    ddm_applicable,
    score_balance_sheet_leverage,
)


def test_fcf_payout_weighted_higher_than_earnings():
    """High FCF stress should score lower than healthy FCF at same earnings payout."""
    good_fcf, _, _ = compute_safety_score(0.50, 0.40, 15, 0.8)
    bad_fcf, _, _ = compute_safety_score(0.50, 1.10, 15, 0.8)
    assert good_fcf is not None and bad_fcf is not None
    assert bad_fcf < good_fcf


def test_ddm_gating_low_yield_and_high_fcf_payout():
    ok, _ = ddm_applicable(0.015, 0.30)
    assert not ok
    ok, _ = ddm_applicable(0.05, 0.95)
    assert not ok
    ok, _ = ddm_applicable(0.04, 0.45)
    assert ok


def test_reverse_dcf_near_forward_growth():
    fcf_ps = 10.0
    growth = 0.08
    fv, _ = compute_dcf(fcf_ps, growth, None)
    assert fv is not None
    implied = compute_reverse_dcf(fcf_ps, fv)
    assert implied is not None
    assert abs(implied - growth) < 0.005


def test_negative_equity_uses_net_debt_over_fcf():
    """MO-style negative book equity should score via Net Debt/FCF, not skip leverage."""
    result = score_balance_sheet_leverage(
        -7.4,
        stockholders_equity=-3_452_000_000,
        net_debt=21_235_000_000,
        free_cash_flow=9_074_000_000,
    )
    assert result is not None
    score, category, note = result
    assert category == "Net Debt / FCF"
    assert "Negative book equity" in note
    assert score == 55  # ~2.3x net debt / FCF → elevated but manageable band

    overall, _, rows = compute_safety_score(
        1.01, 0.77, 16, -7.4,
        stockholders_equity=-3_452_000_000,
        net_debt=21_235_000_000,
        free_cash_flow=9_074_000_000,
    )
    cats = [r["Category"] for r in rows]
    assert "Net Debt / FCF" in cats
    assert overall is not None
    assert overall > 44  # leverage component now included vs pre-Phase-4 skip
