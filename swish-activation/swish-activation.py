import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x)

    sigma = 1.0 / (1 + np.exp(-x))
    swish = x * sigma

    return swish