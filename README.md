# Quant Foundations

Weekly self-study projects to build quant fundamentals through **theory + implementation**:
- Monte Carlo
- Estimation
- Confidence intervals
- Convergence

## Skills demonstrated
- Monte Carlo estimators and validation (LLN/CLT intuition in code)
- Standard error + 95% confidence intervals + coverage checks
- Vectorized NumPy workflows and clean script structure (`main()`, functions, reproducibility via seeds)

## Environment
- Python (virtual environment in `.venv/` — gitignored)
- Key libraries: `numpy`, `pandas`, `scipy`, `matplotlib`

## Repository structure
- `week01_monte_carlo/` — Monte Carlo fundamentals:
  - `monte_carlo_pi.py`: π estimation + SE + CI
  - `normal_moments.py`: estimates of E[X], E[X²], Var(X) + SE/CI
  - `convergence_experiments.py`: empirical 1/√N convergence + diagnostic ratio
  - `notes.md`: theory notes (LLN/CLT/CI interpretation)
- `common/` — reusable utilities (WIP)

## Setup
Activate venv (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

## Install dependencies (if needed)
```powershell
pip install -r requirements.txt
```

## Run scripts (Week 1):
From repo root:
```powershell
python .\week01_monte_carlo\monte_carlo_pi.py
python .\week01_monte_carlo\normal_moments.py
python .\week01_monte_carlo\convergence_experiments.py
```

## Sample output (Week 1)

### Monte Carlo π (N=100000)
- π estimate: **3.14412**
- SE(π): **0.00519**
- 95% CI: **[3.13395, 3.15429]** (contains true π)

### Normal moments (X ~ N(0,1), N=100000)
- E[X] ≈ **0.00097**, SE ≈ **0.00317**, CI contains **0**
- E[X²] ≈ **1.00180**, SE ≈ **0.00447**, CI contains **1**

### Convergence check (E[X²])
- n = 10³ → SE ≈ 0.0437  
- n = 10⁴ → SE ≈ 0.0143  
- n = 10⁵ → SE ≈ 0.00446  
- n = 10⁶ → SE ≈ 0.00141  
(consistent with **SE ∝ 1/√N**)