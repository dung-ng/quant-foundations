import numpy as np

def run_normal_moments_experiment(n):
    x = np.random.standard_normal(n)
    mean_x_estimate = np.mean(x)

    x_square = np.square(x)
    mean_x2_estimate = np.mean(x_square)

    variance_estimate_direct = np.var(x)

    se_mean_x = np.sqrt(variance_estimate_direct) / np.sqrt(n)

    se_mean_x2 = np.std(x_square) / np.sqrt(n)

    return mean_x_estimate, se_mean_x, mean_x2_estimate, se_mean_x2

def compute_confidence_interval (estimate, se, z=1.96):

    ci_lower = estimate - z * se
    ci_upper = estimate + z * se

    return ci_lower, ci_upper

def main():
    N = 100000
    seed = 42
    np.random.seed(seed)
    print(f"Running with N = {N}, seed = {seed}")

    mean_x_estimate, se_mean_x, mean_x2_estimate, se_mean_x2 = run_normal_moments_experiment(N)

    ci_x_lower, ci_x_upper = compute_confidence_interval(mean_x_estimate, se_mean_x)
    ci_x2_lower, ci_x2_upper = compute_confidence_interval(mean_x2_estimate, se_mean_x2)

    print(f"Mean E[X] is: {mean_x_estimate}")
    print(f"Standard Error E[X] is: {se_mean_x}")
    print(f"Mean E[X^2] is: {mean_x2_estimate}")
    print(f"Standard Error E[X^2] is: {se_mean_x2}")
    print(f"Confidence Interval of E[X] lower bound is: {ci_x_lower}")
    print(f"Confidence Interval of E[X] upper bound is: {ci_x_upper}")
    if ci_x_lower <= 0 <= ci_x_upper:
        print(f"True value 0 lies in the Confidence Interval of E[X].")
    else:
        print(f"True value 0 does not lie in the Confidence Interval of E[X].")

    print(f"Confidence Interval of E[X^2] lower bound is: {ci_x2_lower}")
    print(f"Confidence Interval of E[X^2] upper bound is: {ci_x2_upper}")
    if ci_x2_lower <= 1 <= ci_x2_upper:
        print(f"True value 1 lies in the Confidence Interval of E[X^2].")
    else:
        print(f"True value 1 does not lie in the Confidence Interval of E[X^2].")

if __name__ == "__main__":
    main()