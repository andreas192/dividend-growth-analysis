"""
Shared configuration and formatting helpers for the dividend-growth analysis suite.

*** Change TICKER (and other settings) here — all notebooks import from this file. ***
"""

# ── Primary analysis target ──────────────────────────────────────────────────
TICKER = "ZTS"          # Change to any US ticker, e.g. "MSFT", "AAPL", "KO"

# ── Valuation assumptions (used in nb_05 and nb_06) ─────────────────────────
REQUIRED_RETURN = 0.09          # Discount rate / required annual return (9%)
TERMINAL_GROWTH_RATE = 0.03     # Long-term perpetual growth rate (3%)
PROJECTION_YEARS = 30           # Years to project forward for YOC / DRIP

# ── Dividend growth scenario assumptions ────────────────────────────────────
GROWTH_SCENARIOS = {
    "Conservative": 0.04,   # 4% annual dividend growth
    "Base":         0.07,   # 7% annual dividend growth
    "Aggressive":   0.11,   # 11% annual dividend growth
}

# ── Display helpers ──────────────────────────────────────────────────────────
def fmt_millions(val: float) -> str:
    """Format a raw dollar value (in absolute $) to $XM / $XB."""
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


def fmt_pct(val: float) -> str:
    """Format a decimal ratio as a percentage string."""
    return f"{val*100:.1f}%"


def cagr(start: float, end: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start <= 0 or years <= 0:
        return float("nan")
    return (end / start) ** (1 / years) - 1
