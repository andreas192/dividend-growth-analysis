import pandas as pd

years = 10
path_to_prices = 'data/prices'

def fetch_stooq_history(sym: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("Date").set_index("Date")
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff].reset_index()
    return df


def update_prices(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    latest_date = pd.Timestamp(df["Date"].max())
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(days=1)
    
    if latest_date >= cutoff:
        return df
    
    new_df = fetch_stooq_history(sym)
    return pd.concat([df, new_df[new_df["Date"] > latest_date]]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

def download_and_store_prices(symbol: str) -> None:
    if not pd.io.common.file_exists(f"{path_to_prices}/{symbol}.csv"):
        prices = fetch_stooq_history(symbol)
        store_prices_to_csv(prices, f'{path_to_prices}/{symbol}.csv')
    else:
        df = pd.read_csv(f"{path_to_prices}/{symbol}.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        prices = update_prices(df, symbol)
        store_prices_to_csv(prices, f'{path_to_prices}/{symbol}.csv')
    
    
def store_prices_to_csv(prices, filename: str = 'prices.csv') -> None:
    """
    Store prices data to a CSV file.
    
    Args:
        prices: DataFrame containing price data
        filename: Output CSV filename
    """
    if prices.empty:
        print("No prices to store")
        return
    
    try:
        prices.to_csv(filename, index=False)
        print(f"Prices successfully stored to {filename}")
    except IOError as e:
        print(f"Error writing to file: {e}")

