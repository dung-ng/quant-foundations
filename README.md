# Quant Foundations

This repository contains my weekly self-study projects to build quant fundamentals through theory and implementation:
   - Monte Carlo
   - Estimation
   - Confidence intervals
   - Convergence

# Skills demonstrated

Monte Carlo estimators and validation (LLN/CLT intuition in code)
Standard error + 95% confidence intervals + coverage checks
Vectorized NumPy workflows and clean script structure (main(), functions, reproducibility via seeds)

## Environment
- Python (venv in `.venv/`)
- Key libraries: numpy, pandas, scipy, matplotlib

## Repository structure
- `week01_monte_carlo/`- Monte Carlo fundamentals:
      + monte_carlo_pi.py: π estimation + SE + CI
      + normal_moments.py: estimates of E[X], E[X²], Var(X) + SE/CI
      + convergence_experiments.py: empirical 1/√N convergence + diagnostic ratio
      + notes.md: theory notes (LLN/CLT/CI interpretation) 
- `common/`: reusable utilities

## How to run
1. Activate venv
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
2. Install dependencies (if needed)
   - `pip install -r requirements.txt`
3. Run scripts (Week 1):
   - From repo root:
      + python .\week01_monte_carlo\monte_carlo_pi.py
      + python .\week01_monte_carlo\normal_moments.py
      + python .\week01_monte_carlo\convergence_experiments.py

   - Expected results (sanity checks):
      + monte_carlo_pi.py: π estimate close to 3.1416 and CI typically contains np.pi
      + normal_moments.py: CI for E[X] contains 0 and CI for E[X²] contains 1
      + convergence_experiments.py: SE decreases ~1/√N as N increases by powers of 10