import math
import numpy as np
from scipy.spatial import distance


def genData(
    num_data: int,
    num_features: int,
    num_clients: int,
    num_vehicules: int = 1,
    deg: int = 1,
    noise_width: float = 0,
    seed: int = 135,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if deg < 1:
        raise ValueError(f"Polynomial degree must be >= 1, got {deg}")
    if noise_width < 0:
        raise ValueError(f"noise_width must be non-negative, got {noise_width}")
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # the number of nodes
    m = num_clients+1
    coords = np.concatenate((rnd.uniform(-2, 2, (m // 2, 2)), rnd.normal(0, 1, (m - m // 2, 2))))
    
    # condensed pairwise distances (upper triangle, row-major)
    dist_matrix = distance.cdist(coords, coords, "euclidean")
    dist_upper = dist_matrix[~np.eye(m, dtype=bool)]

    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m * (m - 1), p)) * rnd.uniform(-2, 2, (m * (m - 1), p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # feature-based cost
    feature_cost = ((x @ B.T / np.sqrt(p) + 3) ** deg) / 3 ** (deg - 1)
    # noise
    noise = rnd.uniform(1 - noise_width, 1 + noise_width, (n, m * (m - 1)))
    # broadcast base distances over data points
    c = dist_upper + feature_cost * noise
    # rounding
    c = np.around(c, decimals=4).astype(np.float32)
    # demands
    demands = np.zeros(m, dtype=int)
    demands[1:] = rnd.randint(1, 10, size=num_clients)
    # capacity
    capacity = math.ceil((np.sum(demands) / num_vehicules) * 1.2)
    
    return x, c, demands, int(capacity)