import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    w = np.ones(len(X[0]))
    b = 0.0
    for _ in range(steps):
        preds = _sigmoid(np.dot(X, w.T) + b)

        loss = -1.0 / len(X) * np.sum([y[i] * np.log(preds[i]) + (1-y[i]) * np.log(1-preds[i]) for i in range(len(y))])
        
        w -= lr * np.dot(X.T, (preds - y))
        b -= lr * np.mean(preds - y)
        
    return (w,b)