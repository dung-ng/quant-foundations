import numpy as np

def single_experiment(n):
    x = np.random.standard_normal(n)

    x_square = np.square(x)
    mean_x2_estimate = np.mean(x_square)

    absolute_error = abs(mean_x2_estimate - 1)
    se_x2 = np.std(x_square) / np.sqrt(n)
    
    diagnostic_ratio = absolute_error / se_x2

    return mean_x2_estimate, absolute_error, se_x2, diagnostic_ratio

def main():
    seed = 42
    np.random.seed(seed)

    n_values = 10 ** np.arange(3, 7)
    print(f"n_values: {n_values}")

    for n in n_values:
        mean_x2_estimate, absolute_error, se_x2, diagnostic_ratio = single_experiment(n)

        print("\n")
        print(f"For n = {n}:")
        print(f"E[X^2] estimate = {mean_x2_estimate}")
        print(f"Absolute error vs 1 = {absolute_error}")
        print(f"Standard error of E[X^2] = {se_x2}")
        print(f"Diagnostic ratio = {diagnostic_ratio}")

if __name__ == "__main__":
    main()