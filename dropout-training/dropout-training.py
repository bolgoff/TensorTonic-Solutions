import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x, dtype=float)
    
    rand_fn = rng.random if rng is not None else np.random.random
    randoms = rand_fn(size=x.shape)
    
    mask = randoms < (1 - p)
    scale = mask.astype(float) / (1 - p)
    
    output = x * scale
    return (output, scale)