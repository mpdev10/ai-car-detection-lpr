import numpy as np


def compute_centroids(boxes):
    centroids = np.vstack(((boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2))
    return centroids.transpose()


def euclidean_distance(p_from, p_to):
    return np.sum(np.abs(p_from - p_to))
