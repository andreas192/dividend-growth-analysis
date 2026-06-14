"""
Optional qualitative risk/opportunity flags (not in SEC data).

Used by nb_04 and scripts/compare_tickers.py for Dividendology-style context.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualitativeFlag:
    topic: str
    severity: str  # "opportunity" | "watch" | "risk"
    detail: str


# Curated flags aligned with Dividendology coverage — extend as needed.
QUALITATIVE_FLAGS: dict[str, list[QualitativeFlag]] = {
    "ASML": [
        QualitativeFlag(
            "EUV monopoly",
            "opportunity",
            "Only supplier of extreme ultraviolet lithography; hyperscaler AI capex is a tailwind.",
        ),
    ],
    "PEP": [
        QualitativeFlag(
            "Snacks vs beverages mix",
            "watch",
            "Volume pressure in North America beverages; FCF coverage is the key dividend signal.",
        ),
    ],
    "UNH": [
        QualitativeFlag(
            "Healthcare selloff",
            "opportunity",
            "Regulatory/Medicare noise drove multiple compression; thesis is quality + payout room.",
        ),
    ],
    "MO": [
        QualitativeFlag(
            "Declining volumes",
            "watch",
            "Cigarette volumes fall ~5% annually; dividend supported by pricing power and buybacks.",
        ),
        QualitativeFlag(
            "Negative book equity",
            "watch",
            "Buybacks drove negative shareholders' equity — use Net Debt/FCF, not D/E, for leverage.",
        ),
    ],
    "LMT": [
        QualitativeFlag(
            "Defense backlog",
            "opportunity",
            "Government contracts and backlog support acyclical cash flows through recessions.",
        ),
    ],
    "UPS": [
        QualitativeFlag(
            "Amazon transition",
            "risk",
            "Amazon insourcing and B2C mix shift compress margins; dividend frozen with FCF payout >100%.",
        ),
        QualitativeFlag(
            "Dividend cut risk",
            "risk",
            "Dividendology flags potential cut/freeze if FCF does not recover — not visible in SEC ratios alone.",
        ),
    ],
    "PFE": [
        QualitativeFlag(
            "Patent cliff",
            "risk",
            "Post-COVID revenue cliff and LOE on key drugs (e.g. Eliquis, Ibrance) pressure FCF vs dividend.",
        ),
        QualitativeFlag(
            "Acquisition debt",
            "risk",
            "Seagen and other deals added leverage; dividend yield may be a value trap.",
        ),
    ],
    "MA": [
        QualitativeFlag(
            "Multiple compression",
            "opportunity",
            "EPS grew ~160% over 5Y while price lagged; P/E at multi-year low vs ~32x historical avg.",
        ),
        QualitativeFlag(
            "Regulation (interest caps)",
            "watch",
            "MA earns network fees, not interest income — bank rate-cap risk is largely indirect.",
        ),
        QualitativeFlag(
            "Stablecoins & agentic commerce",
            "opportunity",
            "More digital transactions and AI-driven commerce still need fraud, settlement, and compliance rails.",
        ),
    ],
}


def get_qualitative_flags(ticker: str) -> list[QualitativeFlag]:
    return list(QUALITATIVE_FLAGS.get(ticker.upper(), []))


def format_qualitative_flags(ticker: str) -> str:
    flags = get_qualitative_flags(ticker)
    if not flags:
        return ""
    lines = []
    for f in flags:
        tag = f.severity.upper()
        lines.append(f"[{tag}] {f.topic}: {f.detail}")
    return "\n".join(lines)
