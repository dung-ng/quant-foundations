import numpy as np

def main():
    N = 100000
    seed = 42
    np.random.seed(seed)
    print(f"Running Monte Carlo pi with N = {N}, seed = {seed}")

    x = np.random.uniform(low=-1, high=1, size=N)
    y = np.random.uniform(low=-1, high=1, size=N)
    print(f"Lengths of x is {len(x)} and y is {len(y)}")

    x_square = np.square(x)
    y_square = np.square(y)
    simulated_point = x_square + y_square

    indicator_variable = simulated_point <= 1
    count_hits = np.sum(indicator_variable)
    print(f"{count_hits} hits are inside the circle.")

    if 0 <= count_hits <= N:
        print("The indicator structure is correct!")
        hit_ratio = count_hits / N
        print(f"Hit ratio is: {hit_ratio}.")

        pi_estimate = 4 * hit_ratio
        print(f"The estimate of pi is: {pi_estimate}")
        
        signed_error = pi_estimate - np.pi
        absolute_error = abs(signed_error)
        print(f"The absolute error vs pi is: {absolute_error}")

        se_hit_ratio = np.std(indicator_variable.astype(int)) / np.sqrt(N)
        print(f"Standard error is: {se_hit_ratio}")

        se_pi = 4 * se_hit_ratio
        print(f"Standard error estimate is: {se_pi}")

        CI_lower = pi_estimate - 1.96 * se_pi
        print(f"Confidence interval lower bound is: {CI_lower}")

        CI_upper = pi_estimate + 1.96 * se_pi
        print(f"Confidence interval upper bound is: {CI_upper}")

        inside_ci = CI_lower <= np.pi <= CI_upper
        if inside_ci:
            print("np.pi lies inside the Confidence Interval.")
        else:
            print("np.pi does not lie inside the Confidence Interval.")

    else:
        print("The indicator structure is wrong!")

if __name__ == "__main__":
    main()