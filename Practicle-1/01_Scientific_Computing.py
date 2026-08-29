
"""
LAB PROGRAM 1
Scientific computing framework using Python, NumPy and Pandas.

Covers:
- vector/matrix operations
- matrix multiplication, inverse, determinant
- linear transformation
- DataFrame creation/manipulation
- statistical computations
- ML-oriented example
"""
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

# ---------------- 1. Vectors ----------------
v = np.array([2, 3, 4])
w = np.array([1, 5, 2])

print("VECTOR OPERATIONS")
print("v =", v)
print("w =", w)
print("v + w =", v + w)
print("v - w =", v - w)
print("2v =", 2*v)
print("Dot product =", np.dot(v, w))
print("||v|| =", np.linalg.norm(v))

# ---------------- 2. Matrices ----------------
A = np.array([[2, 1], [1, 3]], dtype=float)
B = np.array([[4, 2], [0, 5]], dtype=float)

print("\nMATRIX OPERATIONS")
print("A + B =\n", A+B)
print("A @ B =\n", A@B)
print("B @ A =\n", B@A)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))

# Solve Ax=b
b = np.array([5, 7], dtype=float)
x = np.linalg.solve(A, b)
print("Solution of Ax=b:", x)
print("Verification A@x:", A@x)

# ---------------- 3. Linear transformation ----------------
theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
point = np.array([1., 0.])
rotated = R @ point

print("\nLINEAR TRANSFORMATION")
print("Rotation matrix:\n", R)
print("Original point:", point)
print("Rotated point:", rotated)

# ---------------- 4. Pandas ----------------
df = pd.DataFrame({
    "Student": ["A","B","C","D","E"],
    "Hours": [2, 4, 5, 7, 8],
    "Attendance": [70, 80, 75, 90, 95],
    "Marks": [45, 55, 60, 72, 80]
})

print("\nDATAFRAME")
print(df)

print("\nDescriptive statistics:")
print(df[["Hours","Attendance","Marks"]].describe())

df["Pass"] = np.where(df["Marks"] >= 50, 1, 0)
print("\nAfter feature creation:")
print(df)

print("\nCorrelation matrix:")
print(df[["Hours","Attendance","Marks"]].corr())

# ---------------- 5. ML significance ----------------
# Simple matrix form y = Xw
X = np.column_stack([
    np.ones(len(df)),
    df["Hours"].values,
    df["Attendance"].values
])
y = df["Marks"].values

w_ls = np.linalg.pinv(X) @ y
pred = X @ w_ls

print("\nLEAST-SQUARES ML EXAMPLE")
print("Learned parameters [intercept, hours, attendance] =", w_ls)
print("Predicted marks =", pred)
print("MSE =", np.mean((y-pred)**2))

print("\nInterpretation: NumPy provides the mathematical operations, while Pandas handles structured data used by ML pipelines.")
