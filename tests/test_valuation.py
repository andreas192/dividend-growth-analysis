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
