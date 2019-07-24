from object_tracking.centroid import compute_centroids, euclidean_distance, intersect_exists


class StateQualifier:
    """
    Klasa określająca stan samochodów
    """

    def __init__(self, parking_place_box, left_tracking_iterations=260):
        """
        :param parking_place_box: koordynaty miejsca parkingowego na obrazie w postaci [x_min, y_min, x_max, y_max]
        :param left_tracking_iterations: liczba iteracji, przez które będzie śledzony samochód po odjechaniu
        """
        self.parking_place_box = parking_place_box
        self.parked_car_id = -1
        self.last_left_id = -1
        self.left_tracking_iterations = left_tracking_iterations
        self.iteration_counter = -1

    def get_state_dict(self, boxes, ids, prev_boxes, prev_ids):
        """
        Metoda zwraca słownik tłumaczący id samochodu na stan
        :param boxes: bounding box'y samochodów w postaci dwuwymiarowego array'a
        :param ids: identyfikatory samochodów w postaci jednowymiarowego array'a
        :param prev_boxes: bounding box'y z poprzedniej iteracji
        :param prev_ids: identyfikatory z poprzedniej iteracji
        :return: słownik, tłumaczący identyfikator samochodu na stan
        """
        centroids = compute_centroids(boxes)
        prev_centroids = compute_centroids(prev_boxes)
        prev_dict = {}
        for i in range(prev_centroids.shape[0]):
            prev_dict[prev_ids[i]] = prev_centroids[i]

        state_dict = {}
        for i in range(centroids.shape[0]):
            state_dict[ids[i]] = ""
            if ids[i] in prev_dict:
                if euclidean_distance(centroids[i], prev_dict[ids[i]]) > 20:
                    state_dict[ids[i]] = "MOVE"
            if ids[i] == self.last_left_id:
                state_dict[ids[i]] = "LEFT"
            if intersect_exists(boxes[i], self.parking_place_box):
                self.parked_car_id = ids[i]
                state_dict[ids[i]] = "ARRIVED"
            elif ids[i] == self.parked_car_id:
                self.parked_car_id = -1
                self.last_left_id = ids[i]
                self.iteration_counter = self.left_tracking_iterations
            if self.iteration_counter == 0:
                self.last_left_id = -1
            elif self.iteration_counter > 0:
                self.iteration_counter = self.iteration_counter - 1

        return state_dict
