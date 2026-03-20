import yfinance as yf
import pandas as pd
from pathlib import Path
import os

def download_adjusted_close (
        tickers: list[str],
        start: str = "2020-01-01",
        end: str | None = None,
):
    """
    Download adjusted close prices for the given tickers and returna clean, date-aligned DataFrame.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    # yfinance typically returns a column MultiIndex when multiple tickers are used
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers[:1]

    # Ensure columns are in the same order as requested
    prices = prices.reindex(columns=tickers)

    # Clean and align
    prices = prices.sort_index().dropna(how="any")

    # Convert index to datetime explicitly
    prices.index = pd.to_datetime(prices.index)
    prices.columns.name = None
    prices.index.name = "Date"

    return prices

def print_sanity_check (prices: pd.DataFrame) -> None:
    """
    Print basic validation outputs for the aligned price matrix.
    """
    print("\n=== SANITY CHECKS ===")
    print(f"Shape: {prices.shape}")
    print(f"Columns: {list(prices.columns)}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    print("\nMissing values per column:")
    print(prices.isna().sum())

    print("\nFirst 5 rows:")
    print(prices.head())

    print("\nLast 5 rows:")
    print(prices.tail())

def main():
    tickers = ["SPY", "AAPL", "NVDA", "TLT"]

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prices = download_adjusted_close(tickers=tickers, start="2020-01-01")

    # Final guardrail
    if prices.empty:
        raise ValueError("Downloaded price DataFrame is empty.")
    
    if list(prices.columns) != tickers:
        raise ValueError(f"Unexpected columns. Expected {tickers}, got {list(prices.columns)}")
    
    output_path = data_dir / "asset_prices.csv"
    prices.to_csv(output_path, index=True, index_label="Date")

    print_sanity_check(prices)
    print(f"\nSaved aligned price data to: {output_path}")

if __name__ == "__main__":
    main()