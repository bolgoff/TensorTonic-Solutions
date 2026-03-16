import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    if len(X) <= 1 or not isinstance(X[0], list):
        return None
    X = np.array(X)

    mean = np.mean(X, axis=0)
    X_cent = X - mean

    out = 1.0 * np.dot(X_cent.T,X_cent) / (len(X) - 1)
    return out