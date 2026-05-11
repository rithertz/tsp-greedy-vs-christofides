def held_karp_tsp(dist_matrix):
    """
    Dynamic programming exact solution (Held-Karp).
    Returns (tour, length). Only feasible for n <= 12 on typical hardware.
    """
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0, 0], 0.0

    inf = float("inf")
    dp = [[inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    dp[1][0] = 0.0  # start at city 0

    for mask in range(1 << n):
        if not (mask & 1):  # city 0 must be present
            continue
        for last in range(n):
            if not (mask >> last) & 1:
                continue
            if dp[mask][last] == inf:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = dp[mask][last] + dist_matrix[last][nxt]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    full_mask = (1 << n) - 1
    best_last = -1
    best_cost = inf
    for last in range(1, n):
        cost = dp[full_mask][last] + dist_matrix[last][0]
        if cost < best_cost:
            best_cost = cost
            best_last = last

    if best_last == -1:
        raise ValueError("Failed to reconstruct a complete Held-Karp tour")

    # Reconstruct the path backwards from the final city, then reverse it.
    reverse_path = []
    mask = full_mask
    last = best_last
    while last != 0:
        reverse_path.append(last)
        new_last = parent[mask][last]
        mask &= ~(1 << last)
        last = new_last

    tour = [0] + list(reversed(reverse_path)) + [0]
    return tour, best_cost
