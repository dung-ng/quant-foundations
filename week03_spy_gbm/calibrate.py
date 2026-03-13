import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("week03_spy_gbm")/"data"/"spy.csv"
data_spy = pd.read_csv(csv_path,parse_dates=["Date"])
data_spy = data_spy.set_index("Date").sort_index()
prices = data_spy["Adj Close"]
log_returns = np.log(prices).diff().dropna()

print(f"number of return rows is: {len(log_returns)}")
print(f"number of price rows is: {len(prices)}")
print(f"min daily return is: {log_returns.min()}")
print(f"max daily return is: {log_returns.max()}")

mu_daily = log_returns.mean()
print(f"daily mean is: {mu_daily}")

mu_ann = mu_daily * 252
print(f"annualized mean is: {mu_ann}")
sigma_daily = log_returns.std()
print(f"daily sigma is: {sigma_daily}")
sigma_ann = sigma_daily * np.sqrt(252)
print(f"annualized sigma is: {sigma_ann}")