# 1. What LLN says in your own words?
Law of Large Numbers says as N increases, the sample averages converge closer to the expected value. This justifies Monte Carlo convergence.

# 2. What CLT adds on top of LLN?
Central Limit Theorem says as N increases, the distribution of sample averages become approximately normal. This is what allows confidence intervals and error bars in simulation
# OR a refined answer:
the sampling distribution of the estimator (sample average) becomes approximately normal (for large N, under finite variance assumptions)

# 3. What a 95% confidence interval means (the correct interpretation)?
95% confidence interval means that if the simulation is processed many times, 95% of the constructed intervals would contain the true value and 5% of the time, the interval will not contain the true value.