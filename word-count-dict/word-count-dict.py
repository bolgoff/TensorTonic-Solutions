def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    counts = {}
    for sent in sentences:
        for word in sent:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1

    return counts