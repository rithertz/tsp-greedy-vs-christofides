import numpy as np

def generate_points_random(n, seed):
    """n points uniformly at random in [0,1]^2."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))

def generate_points_clustered(n, n_clusters=3, cluster_std=0.05, seed=None):
    """
    n points clustered around n_clusters random centers.
    Each point is drawn from a Gaussian around a randomly chosen center,
    then clipped to [0,1].
    """
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, 2))
    points = []
    for _ in range(n):
        c = rng.integers(n_clusters)
        pt = centers[c] + cluster_std * rng.standard_normal(2)
        pt = np.clip(pt, 0, 1)
        points.append(pt)
    return np.array(points)

def distance_matrix(points):
    """Euclidean distance matrix (n x n)."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            D[i, j] = d
            D[j, i] = d
    return D