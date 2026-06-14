"""
Shared dividend safety and valuation helpers (nb_03, nb_05, compare_tickers).
"""
from __future__ import annotations

from utils.config import REQUIRED_RETURN, TERMINAL_GROWTH_RATE


def _score_earnings_payout(pr: float) -> tuple[int, str]:
    if pr < 0.40:
        return 100, "Very Low (strong coverage)"
    if pr < 0.60:
        return 85, "Low (solid coverage)"
    if pr < 0.75:
        return 65, "Moderate"
    if pr < 1.0:
        return 40, "High (caution)"
    return 10, "Exceeds earnings!"


def _score_fcf_payout(v: float) -> tuple[int, str]:
    if v < 0.50:
        return 100, "Well covered by FCF"
    if v < 0.75:
        return 75, "Adequately covered"
    if v < 1.0:
        return 45, "Tight FCF coverage"
    return 10, "Exceeds FCF!"


def _score_debt_to_equity(v: float) -> tuple[int, str]:
    if v < 0.5:
        return 100, "Very low leverage"
    if v < 1.0:
        return 80, "Moderate leverage"
    if v < 2.0:
        return 55, "Elevated leverage"
    return 25, "High leverage"


def _score_net_debt_to_cash_flow(ratio: float) -> tuple[int, str]:
    if ratio < 1.0:
        return 100, f"Net debt {ratio:.1f}x FCF — low leverage"
    if ratio < 2.0:
        return 80, f"Net debt {ratio:.1f}x FCF — manageable"
    if ratio < 3.0:
        return 55, f"Net debt {ratio:.1f}x FCF — elevated"
    return 25, f"Net debt {ratio:.1f}x FCF — high leverage"


def score_balance_sheet_leverage(
    debt_to_equity: float | None,
    *,
    stockholders_equity: float | None = None,
    net_debt: float | None = None,
    free_cash_flow: float | None = None,
    operating_cash_flow: float | None = None,
) -> tuple[int, str, str] | None:
    """
    Score balance-sheet leverage for dividend safety.

    When book equity is zero or negative (e.g. MO after buybacks), D/E is not
    meaningful — fall back to Net Debt / FCF (or OCF).
    """
    negative_equity = (
        stockholders_equity is not None and stockholders_equity <= 0
    ) or (debt_to_equity is not None and debt_to_equity < 0)

    if negative_equity:
        cash_flow = None
        cf_label = "FCF"
        if free_cash_flow is not None and free_cash_flow > 0:
            cash_flow = free_cash_flow
        elif operating_cash_flow is not None and operating_cash_flow > 0:
            cash_flow = operating_cash_flow
            cf_label = "OCF"
        if net_debt is not None and cash_flow is not None and cash_flow > 0:
            ratio = net_debt / cash_flow
            score, note = _score_net_debt_to_cash_flow(ratio)
            note = note.replace("FCF", cf_label)
            prefix = "Negative book equity — "
            return score, f"Net Debt / {cf_label}", prefix + note
        return None

    if debt_to_equity is not None and debt_to_equity > 0:
        score, note = _score_debt_to_equity(debt_to_equity)
        return score, "Debt/Equity", note
    return None


def _score_streak(streak: int) -> tuple[int, str]:
    if streak >= 25:
        return 100, f"{streak} consecutive years"
    if streak >= 10:
        return 75, f"{streak} consecutive years"
    if streak >= 5:
        return 50, f"{streak} consecutive years"
    if streak > 0:
        return 25, f"{streak} consecutive years"
    return 0, "No streak"


def compute_safety_score(
    earnings_payout: float | None,
    fcf_payout: float | None,
    streak: int,
    debt_to_equity: float | None,
    *,
    stockholders_equity: float | None = None,
    net_debt: float | None = None,
    free_cash_flow: float | None = None,
    operating_cash_flow: float | None = None,
) -> tuple[int | None, str, list[dict]]:
    """
    Dividend safety score with FCF payout weighted >= earnings payout.

    Returns (overall_score, label, detail_rows) where detail_rows is a list of
    dicts with keys Category, Score, Weight, Assessment.
    """
    weighted: list[tuple[int, int, str, str]] = []

    if fcf_payout is not None and fcf_payout > 0:
        score, note = _score_fcf_payout(fcf_payout)
        weighted.append((score, 40, "FCF Payout", note))
    elif earnings_payout is not None and earnings_payout > 0:
        score, note = _score_earnings_payout(earnings_payout)
        weighted.append((score, 40, "Earnings Payout", note))
    else:
        weighted.append((0, 0, "FCF Payout", "Missing — check cash-flow coverage"))

    if fcf_payout is not None and fcf_payout > 0 and earnings_payout is not None and earnings_payout > 0:
        score, note = _score_earnings_payout(earnings_payout)
        weighted.append((score, 20, "Earnings Payout", note))

    leverage = score_balance_sheet_leverage(
        debt_to_equity,
        stockholders_equity=stockholders_equity,
        net_debt=net_debt,
        free_cash_flow=free_cash_flow,
        operating_cash_flow=operating_cash_flow,
    )
    if leverage is not None:
        score, category, note = leverage
        weighted.append((score, 20, category, note))

    if streak > 0:
        score, note = _score_streak(streak)
        weighted.append((score, 20, "Growth Streak", note))

    active = [(s, w, cat, note) for s, w, cat, note in weighted if w > 0]
    if not active:
        return None, "N/A", []

    total_w = sum(w for _, w, _, _ in active)
    overall = int(round(sum(s * w for s, w, _, _ in active) / total_w))
    label = (
        "VERY SAFE" if overall >= 80
        else "SAFE" if overall >= 60
        else "MODERATE" if overall >= 40
        else "RISKY"
    )

    rows = [
        {"Category": cat, "Score": score, "Weight": f"{weight}%", "Assessment": note}
        for score, weight, cat, note in active
    ]
    rows.append({"Category": "── OVERALL ──", "Score": overall, "Weight": "100%", "Assessment": label})
    return overall, label, rows


def ddm_applicable(
    div_yield: float | None,
    fcf_payout: float | None,
    *,
    min_yield: float = 0.02,
    max_fcf_payout: float = 0.90,
) -> tuple[bool, str]:
    """Gate Gordon DDM when yield is too low or FCF coverage is stressed."""
    if div_yield is not None and div_yield < min_yield:
        return False, f"Yield below {min_yield:.0%} — use reverse DCF / growth framing instead"
    if fcf_payout is not None and fcf_payout > max_fcf_payout:
        return False, f"FCF payout above {max_fcf_payout:.0%} — dividend may be at risk"
    return True, ""


def compute_ddm(latest_dps: float, div_growth: float, price: float | None) -> tuple[float | None, float | None]:
    g = max(0.01, min(div_growth, 0.20))
    if REQUIRED_RETURN <= g:
        return None, None
    d1 = latest_dps * (1 + g)
    fv = d1 / (REQUIRED_RETURN - g)
    mos = (fv - price) / price if price else None
    return fv, mos


def compute_dcf(
    fcf_ps: float,
    fcf_growth: float,
    price: float | None,
    *,
    years: int = 10,
    r: float | None = None,
    terminal_growth: float | None = None,
    clamp_growth: bool = True,
) -> tuple[float | None, float | None]:
    g = fcf_growth
    if clamp_growth:
        g = max(0.01, min(fcf_growth, 0.25))
    discount = r if r is not None else REQUIRED_RETURN
    tg = terminal_growth if terminal_growth is not None else TERMINAL_GROWTH_RATE
    if discount <= tg:
        return None, None
    pv = 0.0
    fcf_n = fcf_ps
    for t in range(1, years + 1):
        fcf_n *= 1 + g
        pv += fcf_n / ((1 + discount) ** t)
    tv = fcf_n * (1 + tg) / (discount - tg)
    fv = pv + tv / ((1 + discount) ** years)
    mos = (fv - price) / price if price else None
    return fv, mos


def compute_reverse_dcf(
    fcf_ps: float,
    price: float,
    *,
    years: int = 10,
    r: float | None = None,
    terminal_growth: float | None = None,
    g_min: float = 0.0,
    g_max: float = 0.30,
) -> float | None:
    """Implied FCF growth rate baked into the current price (binary search)."""
    if fcf_ps <= 0 or price <= 0:
        return None
    discount = r if r is not None else REQUIRED_RETURN
    tg = terminal_growth if terminal_growth is not None else TERMINAL_GROWTH_RATE
    if discount <= tg:
        return None

    def fair_value(growth: float) -> float:
        fv, _ = compute_dcf(
            fcf_ps, growth, None, years=years, r=discount, terminal_growth=tg, clamp_growth=False
        )
        return fv or 0.0

    lo, hi = g_min, g_max
    if fair_value(hi) < price:
        return hi
    if fair_value(lo) > price:
        return lo

    for _ in range(64):
        mid = (lo + hi) / 2
        if fair_value(mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
