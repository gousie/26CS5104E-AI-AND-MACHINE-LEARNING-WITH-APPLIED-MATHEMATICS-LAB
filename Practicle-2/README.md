# LAB PROGRAM 2 — Probability Distributions, Bayes Theorem, MLE and MAP Estimation

## 📌 Description

This lab program demonstrates important concepts in **Probability and Statistics for Machine Learning** using Python.

The program covers:

* Probability distributions
* Binomial distribution
* Normal distribution
* Bayes theorem
* Maximum Likelihood Estimation (MLE)
* Maximum A Posteriori (MAP) estimation
* Beta prior and posterior distributions
* Probability and likelihood visualization

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy**
* **Matplotlib**
* **SciPy**

### Required Libraries

Install the required libraries using:

```bash
pip install numpy matplotlib scipy
```

# 🔹 1. Probability Distributions

The program demonstrates two important probability distributions:

1. Binomial Distribution
2. Standard Normal Distribution

---

## 🔹 1.1 Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent trials.

The program uses:

```python
x = np.arange(0, 11)
binomial_probs = binom.pmf(x, n=10, p=0.3)
```

Parameters:

```text
n = 10       → Number of trials
p = 0.3      → Probability of success
```

The probability mass function is calculated using:

```python
binom.pmf(x, n=10, p=0.3)
```

A bar graph is generated to visualize the distribution.

### Formula

```text
P(X = k) = C(n,k) p^k (1-p)^(n-k)
```

---

# 🔹 1.2 Standard Normal Distribution

The program generates a Standard Normal distribution using:

```python
z = np.linspace(-4, 4, 400)
plt.plot(z, norm.pdf(z, 0, 1))
```

Parameters:

```text
Mean (μ) = 0
Standard deviation (σ) = 1
```

The probability density function is:

```python
norm.pdf(z, 0, 1)
```

The resulting graph shows the familiar bell-shaped normal distribution.

---

# 🔹 2. Bayes Theorem

Bayes theorem is demonstrated using a **fraud detection example**.

The program defines:

```python
P_F = 0.01
P_A_given_F = 0.95
P_A_given_notF = 0.05
```

Where:

```text
P(F)          = Probability of fraud
P(A | F)      = Probability of alert given fraud
P(A | ¬F)     = Probability of alert given no fraud
```

First, the probability of an alert is calculated:

```python
P_A = P_A_given_F*P_F + P_A_given_notF*(1-P_F)
```

Then Bayes theorem is used:

```python
P_F_given_A = P_A_given_F*P_F/P_A
```

### Bayes Theorem Formula

```text
P(F | A) = P(A | F) P(F) / P(A)
```

This demonstrates an important concept:

> Even when a test or alert is highly accurate, the actual probability of the condition after a positive result can be much lower because of the base rate.

---

# 🔹 3. Maximum Likelihood Estimation (MLE)

MLE is used to estimate an unknown parameter based on observed data.

The program uses a **Bernoulli distribution** with:

```python
data = np.array([1,1,1,1,1,1,1,0,0,0])
```

There are:

```text
Successes = 7
Failures  = 3
Total     = 10
```

The MLE of the Bernoulli probability `p` is the sample mean:

```python
p_mle = data.mean()
```

Therefore:

```text
p_MLE = 7 / 10
      = 0.7
```

---

## 🔹 Likelihood Verification

The program also evaluates the likelihood for many possible values of `p`:

```python
p_grid = np.linspace(0.001, 0.999, 500)
```

The likelihood is calculated using:

```python
likelihood = np.array([
    np.prod(p**data * (1-p)**(1-data)) for p in p_grid
])
```

The likelihood curve is plotted to show where the maximum occurs.

The MLE is marked on the graph:

```python
plt.axvline(p_mle, linestyle="--", label="MLE")
```

The peak of the likelihood occurs at approximately:

```text
p = 0.7
```

---

# 🔹 4. MAP Estimation

MAP stands for **Maximum A Posteriori estimation**.

Unlike MLE, MAP estimation incorporates prior knowledge about the parameter.

The program uses a **Beta(2,2) prior**:

```python
alpha, beta_prior = 2, 2
```

Observed data:

```text
Successes = 7
Failures  = 3
```

---

## 🔹 Posterior Distribution

For a Beta prior and Bernoulli/binomial data, the posterior is also a Beta distribution.

The posterior parameters are calculated as:

```python
post_alpha = alpha + successes
post_beta = beta_prior + failures
```

Therefore:

```text
Prior      = Beta(2,2)

Data:
Successes  = 7
Failures   = 3

Posterior   = Beta(9,5)
```

---

# 🔹 5. MAP Estimate

For a Beta distribution with parameters `α` and `β`, when both parameters are greater than 1, the mode is:

```text
MAP = (α - 1) / (α + β - 2)
```

The program calculates:

```python
map_estimate = (post_alpha-1)/(post_alpha+post_beta-2)
```

For the posterior `Beta(9,5)`:

```text
MAP = (9 - 1) / (9 + 5 - 2)

    = 8 / 12

    = 0.6667
```

Therefore:

```text
MAP estimate ≈ 0.6667
```

---

# 🔹 6. Posterior Mean

The posterior mean is calculated using:

```python
posterior_mean = post_alpha/(post_alpha+post_beta)
```

For `Beta(9,5)`:

```text
Posterior Mean = 9 / (9 + 5)
               = 9 / 14
               ≈ 0.6429
```

Therefore:

```text
MAP estimate       ≈ 0.6667
Posterior mean     ≈ 0.6429
```

---

# 📊 Results Summary

| Concept              |    Result |
| -------------------- | --------: |
| Binomial trials      |        10 |
| Binomial probability |       0.3 |
| Fraud probability    |      0.01 |
| P(Alert)             |     0.059 |
| P(Fraud | Alert)     |  ≈ 0.1610 |
| Bernoulli successes  |         7 |
| Bernoulli failures   |         3 |
| MLE of `p`           |       0.7 |
| Prior                | Beta(2,2) |
| Posterior            | Beta(9,5) |
| MAP estimate         |  ≈ 0.6667 |
| Posterior mean       |  ≈ 0.6429 |

---

# 📈 Graphs Generated

The program generates three visualizations:

### 1. Binomial Distribution

Shows the probability of obtaining different numbers of successes in 10 trials.

### 2. Standard Normal Distribution

Shows the probability density of a standard normal random variable.

### 3. Bernoulli Likelihood

Shows how the likelihood changes for different values of `p` and identifies the MLE.

### 4. Posterior Distribution

Shows the Beta posterior distribution and identifies the MAP estimate.

---

# ▶️ How to Run

### Step 1: Check Python Installation

```bash
python --version
```

### Step 2: Install Required Libraries

```bash
pip install numpy matplotlib scipy
```

### Step 3: Navigate to the Program Folder

```bash
cd "LAB PROGRAM 2"
```

### Step 4: Run the Program

```bash
python program2.py
```

The program will display the calculated results in the terminal and open the generated graphs.

---

# 🧠 Important Concepts

| Concept                  | Meaning                                                       |
| ------------------------ | ------------------------------------------------------------- |
| Probability Distribution | Describes possible outcomes and their probabilities           |
| Binomial Distribution    | Models successes in fixed independent trials                  |
| Normal Distribution      | Continuous bell-shaped probability distribution               |
| Bayes Theorem            | Updates probability using new evidence                        |
| MLE                      | Estimates parameters by maximizing likelihood                 |
| MAP                      | Estimates parameters using likelihood and prior information   |
| Prior                    | Belief about a parameter before observing data                |
| Posterior                | Updated belief after observing data                           |
| Likelihood               | Measures how well parameter values explain observed data      |
| Beta Distribution        | Common prior/posterior distribution for Bernoulli probability |

---

# ⏱️ Time and Space Complexity

Let:

* `n` = number of observations
* `m` = number of parameter values evaluated in the likelihood grid

### Binomial Distribution

Calculating probabilities for `m` values is approximately:

```text
O(m)
```

### MLE

Calculating the mean of `n` observations:

```text
O(n)
```

### Likelihood Grid

The program evaluates the likelihood for `m` possible values of `p`, with `n` observations:

```text
O(m × n)
```

For the given program:

```text
m = 500
n = 10
```

### MAP

The posterior parameters and MAP estimate require constant time:

```text
O(1)
```

### Space Complexity

The arrays and likelihood values require approximately:

```text
O(n + m)
```

---

# 🎯 Learning Outcomes

After completing this lab program, the learner should understand:

1. How to generate probability distributions using SciPy.
2. How Binomial and Normal distributions work.
3. How to apply Bayes theorem to real-world problems.
4. How to calculate the MLE of a Bernoulli parameter.
5. How to visualize a likelihood function.
6. How prior knowledge is incorporated into MAP estimation.
7. How a Beta prior produces a Beta posterior for Bernoulli data.
8. The difference between MLE and MAP.
9. The difference between MAP and posterior mean.
10. How probability and statistical estimation concepts are used in machine learning.

---

# 📜 Conclusion

This lab program provides a practical introduction to **probability distributions and statistical parameter estimation using Python**.

NumPy is used for numerical operations, SciPy provides probability distributions, and Matplotlib is used for visualization.

The program connects fundamental probability concepts such as **Bayes theorem, MLE, and MAP estimation** with techniques commonly used in **machine learning and data science**.

---

