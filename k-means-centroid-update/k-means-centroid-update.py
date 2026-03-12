import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []
    n = len(points)

    points = np.array(points)
    clus = np.array([[0] * len(points[0])] * k, dtype=float)
    counts = {}

    for i in range(n):
        clus[assignments[i]] += points[i]
        if assignments[i] not in counts:
            counts[assignments[i]] = 0
        counts[assignments[i]] += 1

    for i in range(len(set(assignments))):
        clus[i] /= counts[i]

    return clus.tolist()