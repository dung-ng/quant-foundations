import numpy as np

def generate_correlated_normals(n, rho, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if not -1 < rho < 1:
        raise ValueError("rho must be strictly between -1 and 1!")

    z1 = np.random.standard_normal(n)
    z2 = np.random.standard_normal(n)

    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    return x1, x2
        

def main():
    n = 100000
    rho = 0.7
    seed = 42

    np.random.seed(seed)

    x1, x2 = generate_correlated_normals(n,rho)
    mean_x1 = np.mean(x1)
    var_x1 = np.var(x1)
    mean_x2 = np.mean(x2)
    var_x2 = np.var(x2)
    corr_estimate = np.corrcoef(x1, x2)[0,1]
    corr_estimate_difference = corr_estimate - rho    
    print(f"x1 shape is: {x1.shape}, mean={mean_x1: .6f}, var={var_x1: .6f}")
    print(f"x2 shape is: {x2.shape}, mean={mean_x2: .6f}, var={var_x2: .6f}")
    print(f"correlation estimate is: {corr_estimate: .6f}, target rho={rho}, diff={corr_estimate_difference: .6f}")
    
if __name__ == "__main__":
    main()