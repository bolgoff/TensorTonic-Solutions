import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    
    _, counts = np.unique(y, return_counts=True)
    
    p = counts / y.size
    
    log_p = np.zeros_like(p)
    mask = p > 0
    log_p[mask] = np.log2(p[mask])
    
    entropy = -np.sum(p * log_p)
    
    return float(entropy)