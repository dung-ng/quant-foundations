from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf

from week06_mc_option_pricer.mc_option_pricer import monte_carlo_call_price

def fetch_price_data(
        ticker: str,
        period: str = "1y",
) -> pd.DataFrame:
    data = yf.download(tickers=ticker, period=period, interval="1d", auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError(f"No data returned from ticker {ticker}.")
    
    return data

def extract_close_series(data: pd.DataFrame) -> pd.Series:
    if "Close" not in data.columns:
        raise ValueError("Close column not found in downloaded data.")

    close_prices = data["Close"].dropna()

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.squeeze("columns")

    if close_prices.empty:
        raise ValueError("Close price series is empty.")

    return close_prices

def extract_adjusted_close_series(data: pd.DataFrame) -> pd.Series:
    if "Adj Close" in data.columns:
        adj_close_prices = data["Adj Close"].dropna()
    elif "Close" in data.columns:
        adj_close_prices = data["Close"].dropna()
    else:
        raise ValueError("Neither Adj Close nor Close column found in downloaded data.")

    if isinstance(adj_close_prices, pd.DataFrame):
        adj_close_prices = adj_close_prices.squeeze("columns")

    if adj_close_prices.empty:
        raise ValueError("Adjusted-close price series is empty.")

    return adj_close_prices

def compute_log_returns(prices: pd.Series) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Log returns are empty after dropping NaNs.")
    
    return log_returns

def annualize_historical_vol(log_returns: pd.Series, trading_days: int = 252) -> float:
    sigma_daily = float(log_returns.std(ddof=1))
    sigma_annual = sigma_daily * math.sqrt(trading_days)
    return sigma_annual

def choose_atm_strike(spot: float, strike_step: int = 5) -> float:
    return float(round(spot / strike_step) * strike_step)

def main() -> None:
    ticker = "SPY"
    r = 0.04
    maturity_days = 30
    T = maturity_days / 365.0
    n_paths = 400_000
    seed = 42

    data = fetch_price_data(ticker=ticker, period="1y")

    close_prices = extract_close_series(data)
    adj_close_prices = extract_adjusted_close_series(data)

    S0 = float(close_prices.iloc[-1])
    log_returns = compute_log_returns(prices=adj_close_prices)
    sigma = annualize_historical_vol(log_returns=log_returns)
    K = choose_atm_strike(S0, strike_step=5)

    result = monte_carlo_call_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        seed=seed,
    )

    intrinsic_value = max(S0 - K, 0.0)
    time_value_mc = result.mc_price - intrinsic_value
    time_value_bs = result.bs_price - intrinsic_value
    moneyness = S0 / K
    ci_half_width = 1.96 * result.standard_error

    print("\n=== Stage B: Real-Data-Style SPY Example ===")
    print(f"Ticker                 : {ticker}")
    print(f"Last observation date  : {close_prices.index[-1].date()}")
    print(f"Spot proxy S0 (Close)         : {S0:.2f}")
    print(f"Chosen strike K        : {K:.2f}")
    print(f"Maturity (days)        : {maturity_days}")
    print(f"Maturity T (years)     : {T:.6f}")
    print(f"Return observations    : {len(log_returns)}")
    print(f"Historical annualized volatility proxy   : {sigma:.4f}")
    print(f"Risk-free rate r       : {r:.4f}")
    print(f"Monte Carlo call price : {result.mc_price:.6f}")
    print(f"Standard error         : {result.standard_error:.6f}")
    print(f"95% CI                 : [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"Black-Scholes price    : {result.bs_price:.6f}")
    print(f"Absolute error         : {result.absolute_error:.6f}")

    if result.ci_lower <= result.bs_price <= result.ci_upper:
        print("Validation check       : Black-Scholes price lies inside the MC confidence interval.")
    else:
        print("Validation check       : Black-Scholes price lies outside the MC confidence interval.")
    
    print(f"Moneyness S0/K         : {moneyness:.6f}")
    print(f"Intrinsic value        : {intrinsic_value:.6f}")
    print(f"MC time value          : {time_value_mc:.6f}")
    print(f"BS time value          : {time_value_bs:.6f}")
    print(f"95% CI half-width      : {ci_half_width:.6f}")

if __name__ == "__main__":
    main()