import numpy as np


def compute_centroids(boxes):
    centroids = np.vstack(((boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2))
    return centroids.transpose()


def euclidean_distance(p_from, p_to):
    return np.sum(np.abs(p_from - p_to))


def intersect_exists(box1, box2):
    return box1[2] >= box2[0] and box1[0] <= box2[2] and box1[3] >= box2[1] and box1[1] <= box2[3]
