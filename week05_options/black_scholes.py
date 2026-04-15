import numpy as np
from scipy.stats import norm
import yfinance as yf

def compute_d1_d2(S0, K, T, r, sigma):
    """Compute the Black-Scholes d1 and d2 terms."""
    if np.any(T <= 0):
        raise ValueError("Maturity must be larger than 0.")
    if np.any(sigma <= 0):
        raise ValueError("Volatility must be higher than 0.")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call_price(S0, K, T, r, sigma):
    """Return the Black-Scholes price of a European call option."""
    d1, d2 = compute_d1_d2(S0, K, T, r, sigma)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bs_put_price(S0, K, T, r, sigma):
    """Return the Black-Scholes price of a European put option."""
    d1, d2 = compute_d1_d2(S0, K, T, r, sigma)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

def fetch_spot_proxy(ticker: str = "SPY", period: str = "1y") -> float:
    """Fetch the latest adjusted close as a proxy for the current spot price."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )

    s0 = float(data["Adj Close"].dropna().iloc[-1])
    return s0

def main():
    """Run baseline Black-Scholes checks and one SPY example."""
    S0 = 100
    K = 100
    T = 1.0
    r = 0.03
    sigma = 0.20

    call_price = bs_call_price(S0, K, T, r, sigma)

    if call_price < 0:
        raise ValueError("Call price must be non-negative.")

    put_price = bs_put_price(S0, K, T, r, sigma)

    if put_price < 0:
        raise ValueError("Put price must be non-negative.")
    
    print("\nCall option price is:")
    print(f"{call_price:.2f}")

    print("\nPut option price is:")
    print(f"{put_price:.2f}")

    sigma_array = np.array([0.10, 0.20, 0.30])
    call_price_diff_vol = bs_call_price(S0, K, T, r, sigma_array)

    print(f"\nCall option price with sigma {sigma_array[0]:.2f} is:")
    print(f"{call_price_diff_vol[0]:.2f}")

    print(f"\nCall option price with sigma {sigma_array[1]:.2f} is:")
    print(f"{call_price_diff_vol[1]:.2f}")

    print(f"\nCall option price with sigma {sigma_array[2]:.2f} is:")
    print(f"{call_price_diff_vol[2]:.2f}")

    T_array = np.array([0.5, 1.0, 2.0])
    call_price_diff_T = bs_call_price(S0, K, T_array, r, sigma)

    print(f"\nCall option price with maturity {T_array[0]} is:")
    print(f"{call_price_diff_T[0]:.2f}")

    print(f"\nCall option price with maturity {T_array[1]} is:")
    print(f"{call_price_diff_T[1]:.2f}")

    print(f"\nCall option price with maturity {T_array[2]} is:")
    print(f"{call_price_diff_T[2]:.2f}")

    diff_call_put_price = call_price - put_price
    net_present_value = S0 - K * np.exp(-r * T)
    parity_gap = abs(diff_call_put_price - net_present_value)

    print("\nLHS of put-call parity (C - P):")
    print(round(diff_call_put_price, 2))

    print("\nNet present value is:")
    print(round(net_present_value, 2))

    print("\nPut-Call parity gap is:")
    print(f"{parity_gap:.8f}")

    s0 = fetch_spot_proxy("SPY")
    K_SPY = round(s0 / 5) * 5
    T_SPY = 63 / 252
    r_SPY = 0.04
    sigma_SPY = 0.20

    print("\nLatest SPY adjusted close used as S0:")
    print(f"{s0:.2f}")

    print("\nChosen SPY strike:")
    print(f"{K_SPY:.2f}")

    call_price_SPY = bs_call_price(s0, K_SPY, T_SPY, r_SPY, sigma_SPY)
    print("\nCall option price of SPY example is:")
    print(f"{call_price_SPY:.2f}")

    put_price_SPY = bs_put_price(s0, K_SPY, T_SPY, r_SPY, sigma_SPY)
    print("\nPut option price of SPY example is:")
    print(f"{put_price_SPY:.2f}")

    print("\nInterpretation:")
    print("This example uses the latest SPY adjusted close as a proxy for S0.")
    print("The strike is chosen near the current spot, so the option is roughly at the money.")
    print("With a longer maturity or higher volatility assumption, both option prices would increase.")
    print("These SPY prices are model prices under an assumed volatility, not necessarily market prices.")

if __name__ == "__main__":
    main()