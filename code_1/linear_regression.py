import numpy as np

# Data
X_raw = np.array([
    [25, 22.0, 9.0],
    [32, 24.0, 8.0],
    [40, 26.5, 6.0],
    [45, 27.0, 5.0],
    [50, 29.0, 4.0],
    [55, 30.5, 3.0],
    [60, 28.0, 2.5],
    [65, 31.0, 2.0],
], dtype=float)

y = np.array([112, 118, 128, 132, 138, 145, 150, 158], dtype=float).reshape(-1, 1)

# Add intercept column
ones = np.ones((X_raw.shape[0], 1))
X = np.hstack([ones, X_raw])  # shape (n, 4)

# OLS closed-form (use pinv for stability; also acceptable to use inv if nonsingular)
beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)   # shape (4, 1)

np.set_printoptions(precision=2, suppress=True)
print(np.linalg.pinv(X.T @ X))
print(X.T @ y)


print("w0, w1, w2, w3 =")
print(beta.ravel())

# Predict new sample
x_new = np.array([1, 52, 27.5, 4.5], dtype=float).reshape(1, -1)
y_pred_new = float(x_new @ beta)
print("Predicted SBP for new sample:", y_pred_new)

# Optional evaluation on training set
y_hat = X @ beta
resid = y - y_hat
mse = float(np.mean(resid**2))
ss_res = float(np.sum(resid**2))
ss_tot = float(np.sum((y - np.mean(y))**2))
r2 = 1 - ss_res / ss_tot

print("MSE:", mse)
print("R^2:", r2)
