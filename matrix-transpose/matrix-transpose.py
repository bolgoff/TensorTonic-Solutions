import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    return np.array([[row[i] for row in A] for i in range(len(A[0]))])