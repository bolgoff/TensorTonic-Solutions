def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    prec = len(set(recommended[:k]) & set(relevant)) / k
    rec = len(set(recommended[:k]) & set(relevant)) / len(set(relevant))

    return [prec, rec]