import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([
    [25, 22.0, 9.0],
    [32, 24.0, 8.0],
    [40, 26.5, 6.0],
    [45, 27.0, 5.0],
    [50, 29.0, 4.0],
    [55, 30.5, 3.0],
    [60, 28.0, 2.5],
    [65, 31.0, 2.0],
], dtype=float)

y = np.array([112, 118, 128, 132, 138, 145, 150, 158], dtype=float)

# Fit
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

print("Intercept (beta0):", model.intercept_)
print("Coefs (beta1, beta2, beta3):", model.coef_)

# Predict new sample
x_new = np.array([[52, 27.5, 4.5]], dtype=float)
y_pred_new = model.predict(x_new)[0]
print("Predicted SBP for new sample:", y_pred_new)

# Metrics
r2 = model.score(X, y)
print("R^2:", r2)
