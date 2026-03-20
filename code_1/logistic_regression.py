import numpy as np

X = np.array([
    [1, 25, 22.0, 9.0],
    [1, 32, 24.0, 8.0],
    [1, 40, 26.5, 6.0],
    [1, 45, 27.0, 5.0],
    [1, 50, 29.0, 4.0],
    [1, 55, 30.5, 3.0],
    [1, 60, 28.0, 2.5],
    [1, 65, 31.0, 2.0],
], dtype=float)

sbp = np.array([112, 118, 128, 132, 138, 145, 150, 158], dtype=float).reshape(-1, 1)

# Label: hypertension if SBP >= 140
t = (sbp >= 140).astype(float)  # shape (8,1), values in {0,1}

# Least Squares Classifier: beta = (X^T X)^(-1) X^T t
beta = np.linalg.pinv(X.T @ X) @ (X.T @ t)  # use pinv for safety

# Scores and predictions
scores = X @ beta
t_hat = (scores >= 0.5).astype(int)

# Confusion matrix elements
t_int = t.astype(int)
TP = int(np.sum((t_hat == 1) & (t_int == 1)))
TN = int(np.sum((t_hat == 0) & (t_int == 0)))
FP = int(np.sum((t_hat == 1) & (t_int == 0)))
FN = int(np.sum((t_hat == 0) & (t_int == 1)))

acc = (TP + TN) / len(t)
print("beta:", beta.ravel())
print("scores:", scores.ravel())
print("pred:", t_hat.ravel())
print("TP TN FP FN:", TP, TN, FP, FN)
print("Accuracy:", acc)

# New sample prediction
x_new = np.array([[1, 52, 27.5, 4.5]], dtype=float)
score_new = float(x_new @ beta)
pred_new = int(score_new >= 0.5)
print("New score:", score_new, "New pred:", pred_new)
