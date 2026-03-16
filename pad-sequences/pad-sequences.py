import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L = 0
    for seq in seqs:
        L = max(L, len(seq))

    if max_len:
        L = max(max_len, L)
        
    for i in range(len(seqs)):
        seqs[i] = (seqs[i] + [pad_value] * (L - len(seqs[i])))[:max_len]

    return seqs