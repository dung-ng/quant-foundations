import yfinance as yf
from pathlib import Path
import pandas as pd

out_path = Path("week03_spy_gbm") / "data" / "spy.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
def main():
    ticker = "SPY"
    data = yf.download(ticker, start="2021-03-09", interval="1d", auto_adjust=False)

    # If columns are multi-indexed (yfinance sometimes does this), flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()  # Date becomes a column
    print(data.shape)
    print(data["Date"].min())
    print(data["Date"].max())
    print(data.head(3))
    data.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()