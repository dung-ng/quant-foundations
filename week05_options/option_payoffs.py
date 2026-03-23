import numpy as np

def call_payoff(spot, strike: float):
    payoff_call =  np.maximum(spot - strike, 0)
    return payoff_call

def put_payoff(spot, strike: float):
    payoff_put = np.maximum(strike - spot, 0)
    return payoff_put

def main():
    ST = np.array([80, 100, 120])
    K = 100

    payoff_call = call_payoff(ST, K)
    print("\nCall option payoff:")
    print(payoff_call)

    payoff_put = put_payoff(ST, K)
    print("\nPut option payoff:")
    print(payoff_put)

    spot_scalar = 120
    print("\nScalar call payoff:")
    print(call_payoff(spot_scalar, K))

    print("\nScalar put payoff:")
    print(put_payoff(spot_scalar, K))

if __name__ == "__main__":
    main()