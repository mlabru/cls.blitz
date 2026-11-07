def match_score(prev, curr):

    dist = centroid_distance(prev, curr)

    size_ratio = min(prev.size, curr.size) / \
                 max(prev.size, curr.size)

    score = (dist * 0.7 - size_ratio * 30.0)

    return score

    