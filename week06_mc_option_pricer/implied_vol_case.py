from __future__ import annotations
import math
from datetime import datetime
from typing import Literal
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

from week05_options.black_scholes import bs_call_price
from week06_mc_option_pricer.mc_option_pricer import monte_carlo_call_price

def fetch_price_data(
        ticker: str,
        period: str = "1y",
) -> pd.DataFrame:
    data = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    
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

def pick_expiry_nearest_target(
        expiries: list[str],
        observation_date: pd.Timestamp,
        target_days: int = 30,
) -> str:
    if not expiries:
        raise ValueError("No option expiries returned.")
    
    obs_date = observation_date.date()

    expiry_distances: list[tuple[int, str]] = []
    for expiry_str in expiries:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        days_to_expiry = (expiry_date - obs_date).days

        if days_to_expiry > 0:
            distance = abs(days_to_expiry - target_days)
            expiry_distances.append((distance, expiry_str))

    if not expiry_distances:
        raise ValueError("No future expiries available.")
    
    expiry_distances.sort(key=lambda x: x[0])
    return expiry_distances[0][1]

def fetch_call_chain(
        ticker: str,
        expiry: str,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()

    if calls.empty:
        raise ValueError(f"No call data returned for {ticker} expiry {expiry}.")
    
    return calls

def choose_nearest_strike_from_chain(
        calls: pd.DataFrame,
        spot: float,
) -> pd.Series:
    calls = calls.copy()
    calls["strike_distance"] = (calls["strike"] - spot).abs()
    selected = calls.sort_values("strike_distance").iloc[0]
    return selected

def choose_market_price(option_row: pd.Series) -> tuple[float, Literal["mid", "lastPrice"]]:
    bid = float(option_row.get("bid", np.nan))
    ask = float(option_row.get("ask", np.nan))
    last_price = float(option_row.get("lastPrice", np.nan))

    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        market_price = 0.5 * (bid + ask)
        return market_price, "mid"
    
    if np.isfinite(last_price) and last_price > 0:
        return last_price, "lastPrice"
    
    raise ValueError("No usable market price found from bid/ask or lastPrice.")

def compute_year_fraction(
        observation_date: pd.Timestamp,
        expiry: str,
) -> float:
    obs_date = observation_date.date()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days = (expiry_date - obs_date).days

    if days <= 0:
        raise ValueError("Expiry must be after the observation date.")
    
    return days / 365.0

def implied_volatility_call(
        market_price: float,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma_lower: float = 1e-6,
        sigma_upper: float = 5.0,
) -> float:
    def pricing_error(sigma: float) -> float:
        return bs_call_price(S0=S0, K=K, T=T, r=r, sigma=sigma) - market_price
    
    lower_error = pricing_error(sigma_lower)
    upper_error = pricing_error(sigma_upper)

    if lower_error * upper_error > 0:
        raise ValueError(
            "Implied vol root is not bracketed. "
            "Check the market price, strike, maturity, or sigma bounds."
        )
    
    iv = brentq(pricing_error, sigma_lower, sigma_upper)
    return float(iv)

def main() -> None:
    ticker = "SPY"
    r = 0.04
    target_days = 30
    n_paths = 400_000
    seed = 42

    data = fetch_price_data(ticker=ticker, period="1y")
    close_prices = extract_close_series(data)

    observation_date = close_prices.index[-1]
    S0 = float(close_prices.iloc[-1])

    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    expiry = pick_expiry_nearest_target(
        expiries=expiries,
        observation_date=observation_date,
        target_days=target_days,
    )

    calls = fetch_call_chain(ticker=ticker, expiry=expiry)
    selected_call = choose_nearest_strike_from_chain(calls=calls, spot=S0)

    K = float(selected_call["strike"])
    market_price, market_price_source = choose_market_price(selected_call)

    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    T = compute_year_fraction(observation_date=observation_date, expiry=expiry)

    lower_bound = max(S0 - K * math.exp(-r * T), 0.0)
    if market_price < lower_bound:
        raise ValueError(
            f"Market price {market_price:.6f} is below the no-arbitrage lower bound {lower_bound:.6f}."
        )

    implied_vol = implied_volatility_call(
        market_price=market_price,
        S0=S0,
        K=K,
        T=T,
        r=r,
        )
    
    result = monte_carlo_call_price(
        S0=S0,
        K=K,
        r=r,
        sigma=implied_vol,
        T=T,
        n_paths=n_paths,
        seed=seed,
    )

    intrinsic_value = max(S0 - K, 0.0)
    time_value_market = market_price - intrinsic_value
    time_value_bs = result.bs_price - intrinsic_value
    time_value_mc = result.mc_price - intrinsic_value
    moneyness = S0 / K
    ci_half_width = 1.96 * result.standard_error

    print("\n=== Stage C: Implied-Volatility SPY Call Example ===")
    print(f"Ticker                 : {ticker}")
    print(f"Observation date       : {observation_date.date()}")
    print(f"Spot proxy S0 (Close)  : {S0:.2f}")
    print(f"Selected expiry        : {expiry}")
    print(f"Maturity T (years)     : {T:.6f}")
    print(f"Selected strike K      : {K:.2f}")
    print(f"Moneyness S0/K         : {moneyness:.6f}")
    print(f"Bid                    : {float(selected_call.get('bid', np.nan)):.4f}")
    print(f"Ask                    : {float(selected_call.get('ask', np.nan)):.4f}")
    print(f"Last price             : {float(selected_call.get('lastPrice', np.nan)):.4f}")
    print(f"Market price source    : {market_price_source}")
    print(f"Market option price    : {market_price:.6f}")
    print(f"Implied volatility     : {implied_vol:.6f}")
    print(f"Intrinsic value        : {intrinsic_value:.6f}")
    print(f"Market time value      : {time_value_market:.6f}")
    print(f"BS price (using IV)    : {result.bs_price:.6f}")
    print(f"MC price (using IV)    : {result.mc_price:.6f}")
    print(f"MC time value          : {time_value_mc:.6f}")
    print(f"BS time value          : {time_value_bs:.6f}")
    print(f"Standard error         : {result.standard_error:.6f}")
    print(f"95% CI                 : [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"95% CI half-width      : {ci_half_width:.6f}")
    print(f"Absolute error vs BS   : {result.absolute_error:.6f}")

    if result.ci_lower <= result.bs_price <= result.ci_upper:
        print("Validation check       : Black-Scholes price lies inside the MC confidence interval.")
    else:
        print("Validation check       : Black-Scholes price lies outside the MC confidence interval.")

if __name__ == "__main__":
    main()