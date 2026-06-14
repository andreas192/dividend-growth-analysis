"""
SEC EDGAR XBRL helpers — fetch quarterly and annual fundamentals for any US ticker.
"""
from __future__ import annotations

import pandas as pd
import requests

SEC_HEADERS = {
    "User-Agent": "dividend-growth-analysis/1.0 (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}
YEARS = 10


# ── CIK resolution ────────────────────────────────────────────────────────────

def get_cik_from_ticker(ticker: str) -> str:
    ticker = ticker.lower()

    txt_url = "https://www.sec.gov/include/ticker.txt"
    response = requests.get(txt_url, headers=SEC_HEADERS, timeout=30)
    if response.ok and response.text.strip():
        for line in response.text.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0].strip().lower() == ticker:
                return f"{int(parts[1].strip()):010d}"

    json_url = "https://www.sec.gov/files/company_tickers_exchange.json"
    response = requests.get(json_url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    for row in payload.get("data", []):
        if len(row) >= 3 and str(row[2]).lower() == ticker:
            return f"{int(row[0]):010d}"

    raise ValueError(f"Ticker not found in SEC mapping: {ticker.upper()}")


# ── Company facts ─────────────────────────────────────────────────────────────

def fetch_sec_companyfacts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


# ── Single concept extraction ─────────────────────────────────────────────────

def extract_quarterly_metric(
    companyfacts: dict,
    concept: str,
    preferred_unit: str = "USD",
) -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    if concept not in facts:
        return pd.DataFrame()

    units = facts[concept].get("units", {})
    if not units:
        return pd.DataFrame()

    records = units.get(preferred_unit) or units[next(iter(units))]
    df = pd.DataFrame(records)
    if df.empty or "end" not in df.columns or "val" not in df.columns:
        return pd.DataFrame()

    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    if "form" in df.columns:
        df = df[df["form"].isin(["10-Q", "10-K"])]
    if "fp" in df.columns:
        df = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4", "FY"])]

    df = df[["end", "val"]].rename(columns={"val": concept})
    df = df.dropna(subset=["end", concept]).copy()
    if df.empty:
        return pd.DataFrame()

    df["quarter"] = df["end"].dt.to_period("Q")
    df = df.sort_values("end").drop_duplicates(subset=["quarter"], keep="last")
    return df[["end", "quarter", concept]]


def extract_first_available_metric(
    companyfacts: dict,
    concept_candidates: list[str],
    preferred_unit: str = "USD",
    min_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str | None]:
    canonical_name = concept_candidates[0]
    candidates = []

    for concept in concept_candidates:
        metric_df = extract_quarterly_metric(companyfacts, concept, preferred_unit=preferred_unit)
        if not metric_df.empty:
            latest_end = metric_df["end"].max()
            candidates.append((metric_df, concept, latest_end))

    if not candidates:
        return pd.DataFrame(), None

    if min_end is not None:
        covered = [item for item in candidates if item[2] >= min_end]
        pool = covered if covered else candidates
    else:
        pool = candidates

    best_df, best_concept, _ = max(pool, key=lambda item: item[2])
    best_df = best_df.rename(columns={best_concept: canonical_name})
    return best_df, best_concept


# ── Full quarterly fundamentals table ─────────────────────────────────────────

# Concept candidates for each logical metric (first entry = canonical name)
CONCEPT_MAP: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "NetIncomeLoss": ["NetIncomeLoss", "ProfitLoss"],
    "EarningsPerShareBasic": ["EarningsPerShareBasic", "EarningsPerShareDiluted"],
    "OperatingCashFlow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "CapExRaw": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "DividendsPerShare": [
        "CommonStockDividendsPerShareCashPaid",
        "CommonStockDividendsPerShareDeclared",
    ],
    "CashAndCashEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "TotalDebt": [
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "DebtCurrent",
    ],
    "StockholdersEquity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "SharesOutstanding": [
        "CommonStockSharesOutstanding",
    ],
}


def build_quarterly_df(ticker: str, years: int = YEARS) -> pd.DataFrame:
    """
    Pull SEC fundamentals for *ticker* and return a merged quarterly DataFrame.
    All monetary columns are in raw dollars (not millions).
    """
    cik = get_cik_from_ticker(ticker)
    facts = fetch_sec_companyfacts(cik)

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    frames: dict[str, pd.DataFrame] = {}
    for canonical, candidates in CONCEPT_MAP.items():
        df, _ = extract_first_available_metric(facts, candidates, min_end=cutoff)
        if not df.empty:
            # extract_first_available_metric renames to candidates[0]; map to our canonical key
            col_in_df = candidates[0]
            if col_in_df in df.columns:
                df = df.rename(columns={col_in_df: canonical})
            if canonical in df.columns:
                frames[canonical] = df.set_index("quarter")[[canonical]]

    if not frames:
        return pd.DataFrame()

    base = None
    for name, df in frames.items():
        if base is None:
            base = df
        else:
            base = base.join(df, how="outer")

    if base is None or base.empty:
        return pd.DataFrame()

    # Re-attach end dates
    end_dates: dict[str, pd.Timestamp] = {}
    for name, df in frames.items():
        df2 = df.copy()
        if "end" in df2.columns:
            end_dates.update(df2["end"].to_dict())

    base = base.reset_index()
    # Rebuild end from quarter
    base["end"] = base["quarter"].apply(lambda q: q.to_timestamp(how="end").normalize())
    base = base[base["end"] >= cutoff].sort_values("end").reset_index(drop=True)

    # Derived metrics
    if "OperatingCashFlow" in base.columns and "CapExRaw" in base.columns:
        base["FreeCashFlow"] = base["OperatingCashFlow"] - base["CapExRaw"].abs()

    if "NetIncomeLoss" in base.columns and "DividendsPerShare" in base.columns:
        base["EarningsPayoutRatio"] = (
            base["DividendsPerShare"] / base["EarningsPerShareBasic"].replace(0, float("nan"))
            if "EarningsPerShareBasic" in base.columns
            else float("nan")
        )

    if "CashAndCashEquivalents" in base.columns and "TotalDebt" in base.columns:
        base["NetDebt"] = base["TotalDebt"].fillna(0) - base["CashAndCashEquivalents"].fillna(0)

    if "TotalDebt" in base.columns and "StockholdersEquity" in base.columns:
        base["DebtToEquity"] = base["TotalDebt"].fillna(0) / base["StockholdersEquity"].replace(0, float("nan"))

    return base
