# LAB PROGRAM 1 — Scientific Computing Framework Using Python, NumPy and Pandas

## 📌 Description

This lab program demonstrates the use of **Python, NumPy, and Pandas** for scientific computing and basic machine-learning-oriented computations.

The program covers:

* Vector operations
* Matrix operations
* Matrix multiplication
* Matrix inverse and determinant
* Solving linear equations
* Linear transformations
* Pandas DataFrame creation and manipulation
* Descriptive statistical computations
* Correlation analysis
* Feature creation
* Least-squares regression using NumPy
* Mean Squared Error (MSE)

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy**
* **Pandas**

### Required Libraries

```bash
pip install numpy pandas
```
---

# 🔹 1. Vector Operations

The program creates two vectors:

```python
v = np.array([2, 3, 4])
w = np.array([1, 5, 2])
```

It performs:

* Vector addition
* Vector subtraction
* Scalar multiplication
* Dot product
* Euclidean norm

Example:

```python
v + w
v - w
2 * v
np.dot(v, w)
np.linalg.norm(v)
```

---

# 🔹 2. Matrix Operations

Two matrices are created:

```python
A = np.array([[2, 1], [1, 3]], dtype=float)
B = np.array([[4, 2], [0, 5]], dtype=float)
```

The program performs:

* Matrix addition
* Matrix multiplication
* Determinant
* Inverse

Matrix multiplication is performed using:

```python
A @ B
```

The determinant is calculated using:

```python
np.linalg.det(A)
```

The inverse is calculated using:

```python
np.linalg.inv(A)
```

---

# 🔹 3. Solving Linear Equations

The program solves the system:

```text
Ax = b
```

using:

```python
x = np.linalg.solve(A, b)
```

The solution is then verified using:

```python
A @ x
```

This demonstrates how NumPy can be used to solve systems of linear equations efficiently.

---

# 🔹 4. Linear Transformation

A **45-degree rotation matrix** is created:

```python
theta = np.deg2rad(45)

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
```

The point:

```text
[1, 0]
```

is transformed using:

```python
rotated = R @ point
```

This demonstrates a basic **linear transformation** using matrix multiplication.

---

# 🔹 5. Pandas DataFrame

A student dataset is created using Pandas:

| Student | Hours | Attendance | Marks |
| ------- | ----: | ---------: | ----: |
| A       |     2 |         70 |    45 |
| B       |     4 |         80 |    55 |
| C       |     5 |         75 |    60 |
| D       |     7 |         90 |    72 |
| E       |     8 |         95 |    80 |

The DataFrame contains:

* Student name
* Study hours
* Attendance
* Marks

---

# 🔹 6. Descriptive Statistics

The program calculates statistical information using:

```python
df[["Hours", "Attendance", "Marks"]].describe()
```

It provides:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

---

# 🔹 7. Feature Creation

A new feature called `Pass` is created:

```python
df["Pass"] = np.where(df["Marks"] >= 50, 1, 0)
```

The rule is:

```text
Marks >= 50  → Pass = 1
Marks < 50   → Pass = 0
```

This demonstrates how new features can be generated from existing data.

---

# 🔹 8. Correlation Analysis

The program calculates the correlation between:

* Study Hours
* Attendance
* Marks

using:

```python
df[["Hours", "Attendance", "Marks"]].corr()
```

Correlation helps identify the relationship between numerical variables.

---

# 🔹 9. Machine Learning-Oriented Example

The program uses the matrix equation:

```text
y = Xw
```

where:

* `X` = input/features
* `w` = learned parameters
* `y` = target/output

The feature matrix contains:

```text
Intercept
Hours
Attendance
```

The parameters are calculated using the **pseudo-inverse**:

```python
w_ls = np.linalg.pinv(X) @ y
```

Predictions are generated using:

```python
pred = X @ w_ls
```

---

# 🔹 10. Mean Squared Error

The model performance is evaluated using **Mean Squared Error (MSE)**:

```python
np.mean((y - pred) ** 2)
```

MSE measures the average squared difference between the actual and predicted values.

A lower MSE generally indicates that predictions are closer to the actual values.

---

# ▶️ How to Run

### Step 1: Install Python

Make sure Python 3 is installed.

Check the installation:

```bash
python --version
```

### Step 2: Install Required Libraries

```bash
pip install numpy pandas
```

### Step 3: Run the Program

Navigate to the folder containing the Python file:

```bash
cd "LAB PROGRAM 1"
```

Then run:

```bash
python program1.py
```

---

# 📊 Main Concepts Demonstrated

| Concept               | Library/Method        |
| --------------------- | --------------------- |
| Vector operations     | NumPy                 |
| Dot product           | `np.dot()`            |
| Vector norm           | `np.linalg.norm()`    |
| Matrix addition       | NumPy                 |
| Matrix multiplication | `@`                   |
| Determinant           | `np.linalg.det()`     |
| Matrix inverse        | `np.linalg.inv()`     |
| Linear equations      | `np.linalg.solve()`   |
| Linear transformation | Matrix multiplication |
| DataFrame             | Pandas                |
| Statistics            | `describe()`          |
| Correlation           | `corr()`              |
| Feature creation      | `np.where()`          |
| Pseudo-inverse        | `np.linalg.pinv()`    |
| Prediction            | Matrix multiplication |
| Model evaluation      | MSE                   |

---

# ⏱️ Time and Space Complexity

For an `n × n` matrix:

| Operation             | Approximate Time Complexity |
| --------------------- | --------------------------: |
| Matrix addition       |                       O(n²) |
| Matrix multiplication |                       O(n³) |
| Matrix determinant    |                       O(n³) |
| Matrix inverse        |                       O(n³) |
| Solving `Ax = b`      |                       O(n³) |
| Vector dot product    |                        O(n) |

The exact performance can depend on the NumPy implementation and underlying numerical libraries.

---

# 🎯 Learning Outcomes

After completing this program, the learner should understand:

1. How NumPy represents vectors and matrices.
2. How to perform basic mathematical operations using NumPy.
3. How to calculate matrix determinant and inverse.
4. How to solve systems of linear equations.
5. How matrix multiplication can represent linear transformations.
6. How Pandas DataFrames store structured data.
7. How to perform basic statistical analysis.
8. How to create new features from existing data.
9. How correlation can be calculated between variables.
10. How matrix operations can be used in a simple ML workflow.
11. How least-squares parameters can be calculated using a pseudo-inverse.
12. How MSE can be used to evaluate predictions.

---

## 📜 Conclusion

This program provides a basic introduction to **scientific computing with Python**. NumPy is used for numerical and matrix-based computations, while Pandas is used for structured data manipulation and statistical analysis.

The final least-squares example demonstrates how these mathematical and data-processing concepts form the foundation of **machine learning pipelines**.

---

