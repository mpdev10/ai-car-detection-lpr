import numpy as np

import object_tracking.centroid as centroid


def assign_ids(boxes, prev=None):
    centroids = centroid.compute_centroids(boxes)
    new = prev
    id = np.empty(centroids.shape[0])
    if prev is not None:
        indices = np.arange(0, prev.shape[0])
        while centroids.size > 0 and indices.size > 0:
            candidates = np.ones((prev.shape[0], 2)) * np.iinfo(int).max
            for i in range(0, indices.shape[0]):
                pos = np.argmin(centroid.euclidean_distance(prev[indices[i]], centroids))
                [dist] = centroid.euclidean_distance(prev[indices[i]], [centroids[pos]])
                candidates[indices[i]] = np.array([pos, dist])
            ind = np.argmin(candidates[:, 1]).astype(int)
            new[ind] = centroids[candidates[ind][0].astype(int)]
            id[ind] = ind
            centroids = np.delete(centroids, candidates[ind][0].astype(int), 0)
            for i in range(0, indices.shape[0]):
                if indices[i] == ind:
                    indices = np.delete(indices, i)
                    break
        if centroids.size > 0:
            new = np.vstack((new, centroids))

        return id
