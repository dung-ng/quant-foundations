from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from week05_options.black_scholes import bs_call_price

@dataclass
class MonteCarloResult:
    mc_price: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    bs_price: float
    absolute_error: float

def simulate_terminal_prices_q(
        S0: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        seed: int = 42,        
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T) * z
    ST = S0 * np.exp(drift + diffusion)
    return ST

def call_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)

def compute_confidence_interval(
        discounted_payoffs: np.ndarray,
        confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    n = len(discounted_payoffs)
    estimate = float(np.mean(discounted_payoffs))
    sample_std = float(np.std(discounted_payoffs, ddof=1))
    standard_error = sample_std / math.sqrt(n)

    z_value = 1.96 if confidence_level == 0.95 else 1.96
    ci_lower = estimate - z_value * standard_error
    ci_upper = estimate + z_value * standard_error
    return estimate, standard_error, ci_lower, ci_upper

def monte_carlo_call_price(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        seed: int = 42,
) -> MonteCarloResult:
    ST = simulate_terminal_prices_q(S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)
    payoffs = call_payoff(ST=ST, K=K)
    discounted_payoffs = math.exp(-r * T) * payoffs

    mc_price, standard_error, ci_lower, ci_upper = compute_confidence_interval(discounted_payoffs)
    bs_price = bs_call_price(S0=S0, K=K, T=T, r=r, sigma=sigma)
    absolute_error = abs(mc_price - bs_price)

    return MonteCarloResult(
        mc_price=mc_price,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bs_price=bs_price,
        absolute_error=absolute_error,
    )

def print_validation_report(result: MonteCarloResult) -> None:
    print("=== Week 6 Baseline Validation: MC vs Black-Scholes ===")
    print(f"Monte Carlo call price : {result.mc_price:.6f}")
    print(f"Standard error         : {result.standard_error:.6f}")
    print(f"95% CI                 : [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"Black-Scholes price    : {result.bs_price:.6f}")
    print(f"Absolute error         : {result.absolute_error:.6f}")

    if result.ci_lower <= result.bs_price <= result.ci_upper:
        print("Validation check: Black-Scholes price lies inside the Monte Carlo confidence interval.")
    else:
        print("Validation check: Black-Scholes price lies outside the Monte Carlo confidence interval.")

def main() -> None:
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    n_paths = 400_000
    seed = 42

    result = monte_carlo_call_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        seed=seed,
    )

    print_validation_report(result)

if __name__ == "__main__":
    main()