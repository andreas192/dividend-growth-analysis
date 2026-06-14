"""Tests for Phase 4 qualitative flags."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.qualitative_flags import get_qualitative_flags  # noqa: E402


def test_pfe_and_ups_have_qualitative_flags():
    pfe = get_qualitative_flags("PFE")
    ups = get_qualitative_flags("UPS")
    assert any("Patent" in f.topic for f in pfe)
    assert any("Amazon" in f.topic for f in ups)


def test_ma_has_qualitative_flags():
    ma = get_qualitative_flags("MA")
    assert any("Stablecoins" in f.topic for f in ma)
    assert any("Multiple compression" in f.topic for f in ma)


def test_unknown_ticker_returns_empty():
    assert get_qualitative_flags("ZZZZ") == []
