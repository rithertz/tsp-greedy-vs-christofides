import numpy as np

def greedy_tsp(dist_matrix, start=0):
    """
    Nearest neighbour heuristic.
    Returns (tour, length) where tour is list of cities (starting and ending at 'start').
    """
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    if not 0 <= start < n:
        raise ValueError("start must be a valid city index")

    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    total = 0.0
    while unvisited:
        # find nearest unvisited city
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        total += dist_matrix[current, next_city]
        unvisited.remove(next_city)
        tour.append(next_city)
        current = next_city
    # return to start
    total += dist_matrix[current, start]
    tour.append(start)
    return tour, total
