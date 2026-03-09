import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if sum(p) != 1.0:
        raise ValueError
    return sum(x[i]*p[i] for i in range(len(x)))