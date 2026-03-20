import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_prices(csv_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"])
    prices = prices.sort_index()

    if prices.empty:
        raise ValueError("Price data is empty.")
    
    if prices.isna().any().any():
        raise ValueError("Price data contains missing values.")
    
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Log return matrix is empty after computation.")
    
    return log_returns

def compute_daily_stats(log_returns: pd.DataFrame) -> dict:
    stats = dict()
    stats['daily_mean'] = log_returns.mean()
    stats['daily_vol'] = log_returns.std()
    stats['daily_cov'] = log_returns.cov()
    stats['daily_corr'] = log_returns.corr()

    return stats

def annualize_stats(stats: dict, trading_days: int = 252) -> dict:
    annualized = stats.copy()
    annualized['annualized_mean'] = stats["daily_mean"] * trading_days
    annualized['annualized_vol'] = stats["daily_vol"] * np.sqrt(trading_days)
    annualized['annualized_cov'] = stats["daily_cov"] * trading_days

    return annualized

def save_outputs(log_returns: pd.DataFrame, data_dir: Path) -> None:
    log_returns.to_csv(data_dir / "log_returns.csv", index=True, index_label="Date")

def print_sanity_checks(prices: pd.DataFrame, log_returns: pd.DataFrame) -> None:
    print("=== SANITY CHECKS ===")
    print(f"Prices shape: {prices.shape}")
    print(f"Log returns shape: {log_returns.shape}")

    print("\nFirst 5 rows of log returns:")
    print(log_returns.head())

    print("\nLast 5 rows of log returns:")
    print(log_returns.tail())

def print_daily_stats(stats: dict) -> None:
    print("\nDaily mean:")
    print(stats["daily_mean"])
    print("\nDaily volatility:") 
    print(stats["daily_vol"])
    print("\nDaily covariance:")
    print(stats["daily_cov"])
    print("\nDaily correlation:")
    print(stats["daily_corr"])

def print_annualized_stats(annualized: dict) -> None:
    print("\nAnnualized mean:")
    print(annualized["annualized_mean"])
    print("\nAnnualized volatility:")
    print(annualized["annualized_vol"])
    print("\nAnnualized covariance:")
    print(annualized["annualized_cov"])

def save_stats_outputs(stats: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats["daily_mean"].to_csv(output_dir / "daily_mean.csv", header=["value"])
    stats["daily_vol"].to_csv(output_dir / "daily_vol.csv", header=["value"])
    stats["daily_cov"].to_csv(output_dir / "daily_cov.csv", index=True)
    stats["daily_corr"].to_csv(output_dir / "daily_corr.csv", index=True)

    stats["annualized_mean"].to_csv(output_dir / "annualized_mean.csv", header=["value"])
    stats["annualized_vol"].to_csv(output_dir / "annualized_vol.csv", header=["value"])
    stats["annualized_cov"].to_csv(output_dir / "annualized_cov.csv", index=True)

def plot_corr_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))

    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Daily Return Correlation Matrix")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def compute_cholesky(corr: pd.DataFrame) -> np.ndarray:
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    
    if list(corr.index) != list(corr.columns):
        raise ValueError("Correlation matrix index and columns must match.")
    
    cholesky_factor = np.linalg.cholesky(corr.values)
    return cholesky_factor

def generate_independent_normals(n_samples: int, n_assets: int, seed: int=42) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("The number of samples must be positive.")
    if n_assets <= 0:
        raise ValueError("The number of assets must be positive.")
    
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_samples, n_assets))
    return z

def generate_correlated_normals(independent_normals: np.ndarray, chol: np.ndarray) -> np.ndarray:
    if chol.shape[0] != chol.shape[1]:
        raise ValueError("Cholesky factor must be square.")
    
    if independent_normals.shape[1] != chol.shape[0]:
        raise ValueError("Dimensions are incompatible for matrix multiplication.")
    
    correlated = independent_normals @ chol.T
    return correlated

def compute_empirical_corr(correlated_normals: np.ndarray, columns: list[str]) -> pd.DataFrame:
    if correlated_normals.ndim != 2:
        raise ValueError("correlated_normals must be a 2D array.")

    if correlated_normals.shape[1] != len(columns):
        raise ValueError("Number of columns does not match number of asset names.")
    
    df = pd.DataFrame(correlated_normals, columns=columns)
    empirical_corr = df.corr()
    return empirical_corr

def print_corr_comparison(target_corr: pd.DataFrame, empirical_corr: pd.DataFrame) -> None:
    if list(target_corr.index) != list(empirical_corr.index) or list(target_corr.columns) != list(empirical_corr.columns):
        raise ValueError("Matrix labels do not match.")
    
    diff = empirical_corr - target_corr

    print("\nTarget correlation matrix:")
    print(target_corr)

    print("\nEmpirical correlation matrix:")
    print(empirical_corr)

    print("\nDifference matrix:")
    print(diff)

def prepare_simulation_inputs(prices: pd.DataFrame, annualized_stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if prices.empty:
        raise ValueError("Price data is empty.")
    
    required_keys = ["annualized_mean", "annualized_vol"]
    for key in required_keys:
        if key not in annualized_stats:
            raise ValueError(f"Missing required key: {key}")

    asset_names = list(prices.columns)

    s0 = prices.iloc[-1].to_numpy()
    mu = annualized_stats["annualized_mean"].reindex(asset_names).to_numpy()
    sigma = annualized_stats["annualized_vol"].reindex(asset_names).to_numpy()

    if np.isnan(mu).any() or np.isnan(sigma).any():
        raise ValueError("mu or sigma contains missing values after alignment.")

    return s0, mu, sigma, asset_names

def simulate_multi_asset_gbm_paths(
        s0: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        corr: pd.DataFrame,
        n_steps: int,
        n_paths: int,
        dt: float = 1/252,
        seed: int = 42
) -> np.ndarray:
    
    if s0.ndim != 1 or mu.ndim != 1 or sigma.ndim != 1:
        raise ValueError("s0, mu, and sigma must all be 1D arrays")
    
    if len(mu) != len(s0) or len(sigma) != len(s0):
        raise ValueError("s0, mu, and sigma must have the same length.")
    
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_assets = len(s0)
    cholesky_factor = np.linalg.cholesky(corr.values)

    paths = np.empty((n_steps + 1, n_paths, n_assets))
    paths[0, :, :] = s0

    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma**2) * dt
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):    
        z = rng.standard_normal(size=(n_paths, n_assets))
        correlated = z @ cholesky_factor.T
        diffusion = sigma * sqrt_dt * correlated
        paths[t + 1, :, :] = paths[t, :, :] * np.exp(drift + diffusion)
    
    return paths

def print_path_sanity_check(paths: np.ndarray, asset_names: list[str]) -> None:
    print("\nPath array shape:")
    print(paths.shape)

    print("\nAsset names:")
    print(asset_names)

    print("\nFirst time slice (t=0) for first 3 paths:")
    print(paths[0, :3, :])

    print("\nLast time slice (T) for first 3 paths:")
    print(paths[-1, :3, :])

    print("\nAll prices strictly positive:")
    print(np.all(paths > 0))

def compute_portfolio_terminal_returns(paths: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if paths.ndim !=3:
        raise ValueError("Paths must be 3D.")
    
    initial_prices = paths[0,:,:]
    terminal_prices = paths[-1,:,:]

    asset_terminal_returns = terminal_prices / initial_prices -1.0

    n_assets = paths.shape[2]
    
    if weights is None:
        weights = np.full(n_assets, 1 / n_assets)

    weights = np.asarray(weights, dtype=float)

    if len(weights) != n_assets:
        raise ValueError("Weights length must match number of assets.")
    
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    
    portfolio_terminal_returns = asset_terminal_returns @ weights

    return asset_terminal_returns, portfolio_terminal_returns

def summarize_portfolio_terminal_returns(portfolio_terminal_returns: np.ndarray) -> dict:
    if portfolio_terminal_returns.ndim != 1:
        raise ValueError("portfolio_terminal_returns must be a 1D array.")
    
    if portfolio_terminal_returns.size == 0:
        raise ValueError("portfolio_terminal_returns must not be empty.")
    summary = {
        "p5": np.percentile(portfolio_terminal_returns, 5),
        "p50": np.percentile(portfolio_terminal_returns, 50),
        "p95": np.percentile(portfolio_terminal_returns, 95),
        "prob_loss": np.mean(portfolio_terminal_returns < 0)
    }

    return summary

def print_portfolio_summary(summary: dict) -> None:
    print("\nPortfolio terminal return summary:")
    print(f"p5: {summary['p5']:.4f}")
    print(f"p50: {summary['p50']:.4f}")
    print(f"p95: {summary['p95']:.4f}")
    print(f"Probability of loss: {summary['prob_loss']:.4f}")

def plot_portfolio_return_histogram(portfolio_terminal_returns: np.ndarray, summary: dict, output_path: Path) -> None:
    if portfolio_terminal_returns.ndim != 1:
        raise ValueError("portfolio_terminal_returns must be a 1D array")
    
    if portfolio_terminal_returns.size == 0:
        raise ValueError("portfolio_terminal_returns must not be empty.")
    
    required_keys = ["p5", "p50", "p95", "prob_loss"]
    for key in required_keys:
        if key not in summary:
            raise ValueError(f"Missing required summary key: {key}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(portfolio_terminal_returns, bins=50, edgecolor="black")

    ax.axvline(0.0, linestyle="--", linewidth=2, label="0")
    ax.axvline(summary["p5"], linestyle="--", linewidth=2, label="p5")
    ax.axvline(summary["p50"], linestyle="--", linewidth=2, label="p50")
    ax.axvline(summary["p95"], linestyle="--", linewidth=2, label="p95")

    ax.set_title("Portfolio Terminal Return Distribution")
    ax.set_xlabel("Terminal portfolio return")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_sample_asset_paths(paths: np.ndarray, asset_names: list[str], output_path: Path, n_plot: int = 20) -> None:
    if paths.ndim != 3:
        raise ValueError("paths must be a 3D array.")
    
    if len(asset_names) != paths.shape[2]:
        raise ValueError("asset_names length must match the number of assets.")
    
    if n_plot <= 0:
        raise ValueError("n_plot must be positive")
    
    n_steps_plus_one, n_paths, n_assets = paths.shape
    n_plot = min(n_plot, n_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex=True)
    axes = axes.flatten()

    time_index = np.arange(n_steps_plus_one)

    for asset_idx, ax in enumerate(axes[:n_assets]):
        for path_idx in range(n_plot):
            ax.plot(time_index, paths[:, path_idx, asset_idx], linewidth=1)

        ax.set_title(asset_names[asset_idx])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Simulated price")

    fig.suptitle("Sample Multi-Asset GBM Paths")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def save_portfolio_summary(summary: dict, output_path: Path) -> None:
    required_keys = ["p5", "p50", "p95", "prob_loss"]
    for key in required_keys:
        if key not in summary:
            raise ValueError(f"Missing required summary key: {key}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        {
            "metric": ["p5", "p50", "p95", "prob_loss"],
            "value": [
                summary["p5"],
                summary["p50"],
                summary["p95"],
                summary["prob_loss"],
            ],
        }   
    )

    summary_df.to_csv(output_path, index=False)

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"

    prices = load_prices(data_dir/ "asset_prices.csv")

    expected_cols = ["SPY", "AAPL", "NVDA", "TLT"]
    if list(prices.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {list(prices.columns)}")

    log_returns = compute_log_returns(prices)
    stats = compute_daily_stats(log_returns)
    annualized = annualize_stats(stats)

    target_corr = stats["daily_corr"]

    chol = compute_cholesky(target_corr)

    z = generate_independent_normals(n_samples=100000, n_assets=len(target_corr.columns), seed=42)

    correlated_normals = generate_correlated_normals(z, chol)

    empirical_corr = compute_empirical_corr(correlated_normals, list(target_corr.columns))

    print_sanity_checks(prices, log_returns)
    print_daily_stats(stats)
    print_annualized_stats(annualized)
    save_outputs(log_returns, data_dir)
    save_stats_outputs(annualized, output_dir)

    plot_corr_heatmap(stats["daily_corr"], output_dir / "corr_heatmap.png")

    print_corr_comparison(target_corr, empirical_corr)

    s0, mu, sigma, asset_names = prepare_simulation_inputs(prices, annualized)

    paths = simulate_multi_asset_gbm_paths(s0, mu, sigma, target_corr, n_steps=63, n_paths=10_000)

    print_path_sanity_check(paths, asset_names)

    asset_terminal_returns, portfolio_terminal_returns = compute_portfolio_terminal_returns(paths)

    summary = summarize_portfolio_terminal_returns(portfolio_terminal_returns)

    print_portfolio_summary(summary)

    plot_portfolio_return_histogram(portfolio_terminal_returns, summary, output_dir / "portfolio_return_hist.png")

    plot_sample_asset_paths(paths, asset_names, output_dir / "sample_asset_paths.png", n_plot=20)

    save_portfolio_summary(summary, output_dir / "portfolio_summary.csv")

if __name__ == "__main__":
    main()