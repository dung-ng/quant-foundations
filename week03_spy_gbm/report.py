from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_prices(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    prices = df["Adj Close"].astype(float)
    return prices

def calibrate_from_prices(prices: pd.Series):
    log_returns = np.log(prices).diff().dropna()
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    mu_ann = mu_daily * 252
    sigma_ann = sigma_daily * np.sqrt(252)
    return mu_ann, sigma_ann

def simulate_gbm_paths(S0: float, mu_ann: float, sigma_ann: float, n_paths: int, n_steps: int, seed: int = 42):
    np.random.seed(seed)
    dt = 1 / 252

    Z = np.random.standard_normal((n_paths,n_steps))
    drift = (mu_ann - 0.5 * sigma_ann**2) * dt
    diffusion = sigma_ann * np.sqrt(dt) * Z
    dlogS = drift + diffusion

    log_paths = np.cumsum(dlogS, axis=1)
    zeros = np.zeros((n_paths,1))
    log_paths_with_0 = np.hstack([zeros, log_paths])

    S_paths = S0 * np.exp(log_paths_with_0)
    return S_paths

def main():
    csv_path =  Path("week03_spy_gbm") / "data" / "spy.csv"
    out_dir = Path("week03_spy_gbm") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(csv_path)
    mu_ann, sigma_ann = calibrate_from_prices(prices)

    S0 = float(prices.iloc[-1])
    T_days = 63
    n_steps = T_days
    n_paths = 10_000
    seed = 42

    S_paths = simulate_gbm_paths(S0, mu_ann, sigma_ann, n_paths, n_steps, seed=seed)
    S_T = S_paths[:,-1]

    #Terminal percentiles + loss probability
    p5, p50, p95 = np.percentile(S_T, [5, 50, 95])
    loss_prob = np.mean(S_T < S0)

    #Max drawdown probability + percentiles
    running_max = np.maximum.accumulate(S_paths, axis=1)
    drawdown = S_paths / running_max - 1
    max_drawdown = drawdown.min(axis=1)
    prob_dd_10 = np.mean(max_drawdown <= -0.10)
    dd_p5, dd_p50, dd_p95 = np.percentile(max_drawdown, [5, 50, 95])

    # ---- Print concise report (for README) ----
    print(f"S0={S0:.2f}, mu_ann={mu_ann:.4f}, sigma_ann={sigma_ann:.4f}, horizon={T_days}d, paths={n_paths}")
    print(f"Terminal percentiles: p5={p5:.2f}, p50={p50:.2f}, p95={p95:.2f}")
    print(f"P(S_T < S0)={loss_prob:.4f}")
    print(f"P(max drawdown <= -10%)={prob_dd_10:.4f}")
    print(f"Max drawdown percentiles: p5={dd_p5:.2f}, p50={dd_p50:.2f}, p95={dd_p95:.2f}")

    # ---- Plot 1: sample paths ----
    n_plot = 30
    plt.figure()
    t = np.arange(n_steps + 1)
    for i in range(n_plot):
        plt.plot(t, S_paths[i, :])
    plt.xlabel("Trading days")
    plt.ylabel("Price")
    plt.title(f"SPY GBM sample paths (T={T_days}d, mu={mu_ann:.2f}, sigma={sigma_ann:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir / "spy_gbm_sample_paths.png", dpi=150)
    plt.close()

    # ---- Plot 2: terminal histogram ----
    plt.figure()
    plt.hist(S_T, bins=60)
    plt.axvline(S0, linestyle="--")
    plt.axvline(p5, linestyle="--")
    plt.axvline(p50, linestyle="--")
    plt.axvline(p95, linestyle="--")
    plt.xlabel("Terminal price")
    plt.ylabel("Frequency")
    plt.title(f"SPY GBM terminal distribution (T={T_days}d)")
    plt.tight_layout()
    plt.savefig(out_dir / "spy_gbm_terminal_hist.png", dpi=150)
    plt.close()

    # ---- Plot 3 (optional): max drawdown histogram ----
    plt.figure()
    plt.hist(max_drawdown, bins=60)
    plt.axvline(-0.10, linestyle="--")
    plt.xlabel("Max drawdown (fraction)")
    plt.ylabel("Frequency")
    plt.title(f"SPY GBM max drawdown distribution (T={T_days}d)")
    plt.tight_layout()
    plt.savefig(out_dir / "spy_gbm_max_drawdown_hist.png", dpi=150)
    plt.close()

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    main()