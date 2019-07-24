import numpy as np

import object_tracking.centroid as centroid


class ObjectTracker:
    """
    Klasa odpowiedzialna za śledzenie obiektów i przyznawanie im identyfikatorów
    """

    def __init__(self, class_dict):
        """
        :param class_dict: słownik tłumaczący liczbę na nazwę klasy
        """
        self.class_dict = class_dict
        self.prev_values = None

    def track(self, class_name, prediction):
        """
        Metoda, która przyznaje identyfikatory śledzonym obiektom i odseparowuje je od reszty
        :param class_name: nazwa etykiety śledzonych obiektów
        :param prediction: wynik metody predict() instancji klasy Predictor
        :return: krotka
        """
        tracked_objects, untracked_objects = self._filter_by_class_name(class_name, prediction)
        boxes, _, _ = tracked_objects
        _, ids = self._assign_ids(boxes)
        self.prev_values = boxes, ids
        return tracked_objects, untracked_objects, ids

    def _filter_by_class_name(self, class_name, prediction):
        """
        Metoda filtrująca obiekty z predykcji po nazwie klasy
        :param class_name: nazwa klasy
        :param prediction: wynik metody predict() instancji klasy Predictor
        :return: krotka (matching, not_matching) gdzie obie wartości to krotki w postaci (boxes, labels, probabilities)
        """
        boxes, labels, probs = prediction
        to_delete = np.array([])
        matching_class_indexes = np.array([])
        for i in range(boxes.shape[0]):
            if self.class_dict[labels[i]] != class_name:
                to_delete = np.append(to_delete, [i])
            else:
                matching_class_indexes = np.append(matching_class_indexes, [i])
        matching = np.delete(boxes, to_delete, 0), \
                   np.delete(labels, to_delete, 0), \
                   np.delete(probs, to_delete, 0)
        not_matching = np.delete(boxes, matching_class_indexes, 0), \
                       np.delete(labels, matching_class_indexes, 0), \
                       np.delete(probs, matching_class_indexes, 0)
        return matching, not_matching

    def _assign_ids(self, boxes):
        """
        Metoda przypisująca danym bounding boxom identyfikatory
        :param boxes: array o kształcie (n, 4) gdzie n to liczba prostokątów
        :return: krotka w postaci (boxes, ids), gdzie ids[i] to identyfikator prostokąta z boxes[i]
        """
        global ids
        centroids = centroid.compute_centroids(boxes)
        if self.prev_values is None:
            ids = np.arange(centroids.shape[0])
        elif self.prev_values is not None:
            prev_boxes, prev_ids = self.prev_values
            prev_centroids = centroid.compute_centroids(prev_boxes)
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

        return boxes, ids
