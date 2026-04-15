# Quant Foundations Portfolio

A hands-on portfolio of weekly quantitative finance projects built to develop core quant skills through **theory, implementation, validation, and interpretation**.

## What this repository shows

* Monte Carlo simulation and statistical estimation
* Confidence intervals and convergence diagnostics
* Correlation and covariance modeling
* Cholesky decomposition for correlated simulation
* Geometric Brownian Motion (GBM) calibration and path simulation
* Single-asset and multi-asset scenario analysis using real market data
* Portfolio-level return distribution analysis
* European option payoff modeling and Black-Scholes pricing
* Real-market-data inputs for option pricing examples
* Monte Carlo European option pricing with confidence intervals
* Black-Scholes benchmarking for Monte Carlo validation
* Implied volatility inversion from real listed option prices

## Technical skills demonstrated

* Python for quantitative modeling
* NumPy and pandas for vectorized analysis
* Monte Carlo methods and simulation validation
* Statistical estimation, standard errors, and confidence intervals
* Correlation/covariance estimation and matrix factorization
* Real-market data ingestion with `yfinance`
* Black-Scholes option pricing and payoff modeling
* Numerical root-finding for implied volatility
* Options-chain handling and contract selection
* Data visualization with `matplotlib`
* Clean script structure, reproducibility, and modular functions

## Project highlights

### Week 1 — Monte Carlo Foundations

Built Monte Carlo estimators for:

* π estimation
* normal moments
* convergence experiments

Key ideas:

* Law of Large Numbers
* Central Limit Theorem
* standard error
* 95% confidence intervals
* empirical `1/sqrt(N)` convergence

### Week 2 — Correlated Normals and Cholesky

Implemented correlated random variable simulation using:

* direct correlation construction
* Cholesky decomposition

Validated:

* empirical mean
* empirical variance
* empirical correlation
* matrix reconstruction `L @ L.T = Σ`

### Week 3 — SPY GBM Scenario Engine

Downloaded SPY data, calibrated GBM parameters from log returns, and simulated **10,000** three-month scenarios.

Outputs included:

* terminal percentiles
* probability of loss
* path-dependent max drawdown probability
* sample path and terminal distribution plots

### Week 4 — Multi-Asset GBM Scenario Engine

Built a correlated multi-asset simulation engine using **SPY, AAPL, NVDA, and TLT**.

Pipeline:

* estimate daily log-return statistics
* annualize mean, volatility, and covariance
* estimate cross-asset correlation
* generate correlated shocks via Cholesky decomposition
* simulate **10,000** correlated GBM paths over **63 trading days**
* evaluate equal-weight portfolio terminal returns

Portfolio summary:

* `p5 = -13.58%`
* `p50 = 4.48%`
* `p95 = 28.04%`
* `Probability of loss = 35.12%`

### Week 5 — Option Payoffs and Black-Scholes Pricing

Built a pricing toolkit for European call and put options, starting from payoff functions and extending to Black-Scholes closed-form pricing.

Pipeline:

* implement call and put payoff functions
* compute Black-Scholes `d1` and `d2`
* price European call and put options
* validate non-negativity of option prices
* test sensitivity to volatility and maturity
* verify put-call parity
* apply the model to a real-market-style SPY example using the latest adjusted close as `S0`

Validation summary:

* baseline call price = `9.41`
* baseline put price = `6.46`
* call price increases with volatility
* call price increases with maturity
* put-call parity gap ≈ `0.00000000`

SPY example:

* latest adjusted close used as `S0`
* strike chosen near spot
* Black-Scholes call and put prices computed under an assumed volatility input

### Week 6 — Monte Carlo European Option Pricing and Implied Volatility Workflow

Built a Monte Carlo pricing engine for European call options and validated it against the Black-Scholes benchmark before extending it to real-data-style and implied-volatility-based cases.

Pipeline:

* simulate terminal prices under risk-neutral GBM
* compute discounted European call payoffs
* estimate Monte Carlo option price
* compute standard error and 95% confidence interval
* benchmark Monte Carlo prices against Black-Scholes
* test convergence by increasing the number of simulation paths
* apply the pricer to a real-data-style SPY case using spot, maturity, and historical volatility
* extend the workflow to a real listed SPY option by extracting a market option price and solving for implied volatility
* reprice the same contract using both Black-Scholes and Monte Carlo under the implied-volatility input

Baseline validation summary:

* `n_paths = 100,000` → MC call price = `10.420541`, BS price = `10.450584`
* 95% CI = `[10.328872, 10.512210]`
* `n_paths = 400,000` → MC call price = `10.454033`, BS price = `10.450584`
* 95% CI = `[10.408300, 10.499766]`
* absolute error at `400,000` paths = `0.003449`

Real-data-style SPY case:

* spot proxy from raw close used as `S0`
* historical volatility estimated from adjusted-close log returns
* near-ATM strike selected from spot level
* example result: MC call price = `12.250332`, BS price = `12.248816`
* 95% CI = `[12.196878, 12.303786]`
* intrinsic value = `0.440002`
* time value ≈ `11.81`

Implied-volatility SPY case:

* selected a real listed SPY call near 30 days to expiry
* used bid-ask midpoint as market price input during U.S. market hours
* inverted Black-Scholes numerically to solve for implied volatility
* midpoint-based market option price = `12.165000`
* implied volatility = `0.137866`
* BS price (using IV) = `12.165000`
* MC price (using IV) = `12.166664`
* 95% CI = `[12.112893, 12.220434]`
* absolute error vs BS = `0.001664`

## Repository structure

* `week01_monte_carlo/`
* `week02_correlated_normals/`
* `week03_spy_gbm/`
* `week04_multi_asset_gbm/`
* `week05_options/`
* `week06_mc_option_pricer/`

## Tools and libraries

* Python
* NumPy
* pandas
* SciPy
* matplotlib
* yfinance

## How to run

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
````

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run project scripts from the repository root. Example:

```powershell
python .\week03_spy_gbm\fetch_data.py
python .\week03_spy_gbm\calibrate.py
python .\week03_spy_gbm\report.py
python .\week05_options\option_payoffs.py
python .\week05_options\black_scholes.py
```

## Why this portfolio

This repository is part of a structured transition into quantitative finance, with each project designed to build practical skill in:

* simulation
* statistical reasoning
* stochastic modeling
* option pricing
* risk analysis
* portfolio interpretation