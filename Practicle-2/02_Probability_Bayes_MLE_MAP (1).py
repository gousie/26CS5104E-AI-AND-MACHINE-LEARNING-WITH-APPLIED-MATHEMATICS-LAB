
"""
LAB PROGRAM 2
Probability distributions, Bayes theorem, MLE and MAP estimation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, beta

# ---------------- 1. Distributions ----------------
x = np.arange(0, 11)
binomial_probs = binom.pmf(x, n=10, p=0.3)

plt.figure()
plt.bar(x, binomial_probs)
plt.xlabel("Number of successes")
plt.ylabel("Probability")
plt.title("Binomial Distribution: n=10, p=0.3")
plt.show()

z = np.linspace(-4, 4, 400)
plt.figure()
plt.plot(z, norm.pdf(z, 0, 1))
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Standard Normal Distribution")
plt.show()

# ---------------- 2. Bayes theorem ----------------
# Fraud example
P_F = 0.01
P_A_given_F = 0.95
P_A_given_notF = 0.05

P_A = P_A_given_F*P_F + P_A_given_notF*(1-P_F)
P_F_given_A = P_A_given_F*P_F/P_A

print("BAYES THEOREM")
print("P(Alert) =", P_A)
print("P(Fraud | Alert) =", P_F_given_A)

# ---------------- 3. MLE ----------------
# Bernoulli: 7 successes out of 10
data = np.array([1,1,1,1,1,1,1,0,0,0])
p_mle = data.mean()

print("\nMLE")
print("Observations:", data)
print("MLE for Bernoulli p =", p_mle)

# Verify by evaluating likelihood on a grid
p_grid = np.linspace(0.001, 0.999, 500)
likelihood = np.array([
    np.prod(p**data * (1-p)**(1-data)) for p in p_grid
])

plt.figure()
plt.plot(p_grid, likelihood)
plt.axvline(p_mle, linestyle="--", label="MLE")
plt.xlabel("p")
plt.ylabel("Likelihood")
plt.legend()
plt.title("Bernoulli Likelihood")
plt.show()

# ---------------- 4. MAP with Beta prior ----------------
# Prior Beta(2,2)
alpha, beta_prior = 2, 2
successes, failures = 7, 3

post_alpha = alpha + successes
post_beta = beta_prior + failures

map_estimate = (post_alpha-1)/(post_alpha+post_beta-2)
posterior_mean = post_alpha/(post_alpha+post_beta)

print("\nMAP")
print("Prior: Beta(2,2)")
print("Posterior: Beta({}, {})".format(post_alpha, post_beta))
print("MAP estimate =", map_estimate)
print("Posterior mean =", posterior_mean)

theta = np.linspace(0.001, 0.999, 500)
plt.figure()
plt.plot(theta, beta.pdf(theta, post_alpha, post_beta))
plt.axvline(map_estimate, linestyle="--", label="MAP")
plt.legend()
plt.xlabel("theta")
plt.ylabel("Posterior density")
plt.title("Posterior Distribution")
plt.show()
