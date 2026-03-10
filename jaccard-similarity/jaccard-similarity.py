def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    if not set_a or not set_b:
        return 0.0
    return len(set(set_a) & set(set_b)) / len(set(set_a) | set(set_b))