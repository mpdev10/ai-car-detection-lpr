import numpy as np

import object_tracking.centroid as centroid


def filter_by_class_name(class_dict, class_name, prediction):
    boxes, labels, probs = prediction
    to_delete = np.array([])
    matching_class_indexes = np.array([])
    for i in range(boxes.shape[0]):
        if class_dict[labels[i]] != class_name:
            to_delete = np.append(to_delete, [i])
        else:
            matching_class_indexes = np.append(matching_class_indexes, [i])
    matching = np.delete(boxes, to_delete, 0), \
               np.delete(labels, to_delete, 0), \
               np.delete(probs, to_delete, 0)
    not_matching = np.delete(boxes, matching_class_indexes, 0), \
                   np.delete(labels, matching_class_indexes, 0), \
                   np.delete(matching_class_indexes, 0)
    return matching, not_matching


def assign_ids(boxes, prev_values=None):
    global ids
    centroids = centroid.compute_centroids(boxes)
    if prev_values is None:
        ids = np.arange(centroids.shape[0])
    elif prev_values is not None:
        prev_centroids, prev_ids = prev_values
        ids = np.ones(centroids.shape[0], dtype=int) * (-1)
        centroids_indexes = np.arange(0, centroids.shape[0])
        for i in range(prev_centroids.shape[0]):
            shortest_distance = centroid.euclidean_distance(prev_centroids[i], centroids[centroids_indexes[0]])
            shortest_distance_centroid_index = centroids_indexes[0]
            index_count = 0
            current_list_index = 0
            for j in centroids_indexes:
                distance = centroid.euclidean_distance(prev_centroids[i], centroids[j])
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_distance_centroid_index = j
                    current_list_index = index_count
                index_count = index_count + 1
            ids[shortest_distance_centroid_index] = prev_ids[i]
            centroids_indexes = np.delete(centroids_indexes, current_list_index)
            if centroids_indexes.size == 0:
                break
        for i in range(ids.shape[0]):
            if ids[i] == -1:
                ids[i] = np.max(ids) + 1 if np.min(np.abs(ids)) <= 0 else np.min(np.abs(ids)) - 1

    return centroids, ids
