from pathlib import Path
import pandas as pd
import numpy as np

csv_path = Path("week03_spy_gbm")/"data"/"spy.csv"
data_spy = pd.read_csv(csv_path,parse_dates=["Date"])
data_spy = data_spy.set_index("Date").sort_index()
prices = data_spy["Adj Close"]
log_returns = np.log(prices).diff().dropna()
mu_ann = log_returns.mean() * 252
print(f"Annualized mu is: {mu_ann}")
sigma_ann = log_returns.std() * np.sqrt(252)
print(f"Annualized sigma is: {sigma_ann}")

S0 = float(prices.iloc[-1])
print(f"S0 is: {S0}")
T_days = 63 # 3months trading day
T = T_days / 252
n_paths = 10000
n_steps = T_days
dt = 1/252

seed = 42
np.random.seed(seed)
Z = np.random.standard_normal((n_paths,n_steps))
drift = (mu_ann - 0.5 * sigma_ann**2) * dt
diffusion = sigma_ann * np.sqrt(dt) * Z
dlogS = drift + diffusion
log_paths = np.cumsum(dlogS,axis=1)
zeros = np.zeros((n_paths,1))
log_paths_with_0 = np.hstack([zeros,log_paths])
S_paths = S0 * np.exp(log_paths_with_0)
print(f"S_Paths shape is: {S_paths.shape}")
print(f"First 3 prices of first path:", S_paths[0,:3])
print(f"Last 3 prices of first path:", S_paths[0,-3:])
print(np.allclose(S_paths[:,0], S0))
print(S_paths[:,-1])

S_T = S_paths[:,-1]
p5, p50, p95 = np.percentile(S_T, [5, 50, 95])
print(f"Terminal price percentiles (T={T_days}d): p5={p5:.2f}, p50={p50:.2f}, p95={p95:.2f}")
loss_prob = np.mean(S_T < S0)
print(f"probability of loss is: {loss_prob}")

running_max = np.maximum.accumulate(S_paths, axis=1)
drawdown = S_paths / running_max - 1
max_drawdown = drawdown.min(axis=1)
prob_dd_10 = np.mean(max_drawdown <= -0.10)
print(f"probability drawdown is: {prob_dd_10}")
p_dd5, p_dd50, p_dd95 = np.percentile(max_drawdown, [5, 50, 95])
print(f"Max drawdown percentiles (T={T_days}d): p5={p_dd5:.2f}, p50={p_dd50:.2f}, p95={p_dd95:.2f}")