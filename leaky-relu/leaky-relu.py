import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    return np.array([xi if xi >= 0 else alpha * xi for xi in x])