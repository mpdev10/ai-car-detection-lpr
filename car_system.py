import numpy as np

from lpr.LPR import LPR
from object_tracking.object_tracking import ObjectTracker
from ssd.predictor import Predictor
from state_determining.car_state import StateQualifier
from state_determining.light_state import determine_light_on


class CarSystem:
    """
    Klasa agregująca wszystkie podzespoły systemu
    """

    def __init__(self,
                 detection_predictor: Predictor, state_qualifier: StateQualifier,
                 car_tracker: ObjectTracker, lpr: LPR, frame_skip, light_level_th, prob_th):
        """
        Konstruktor klasy CarSystem
        :param detection_predictor: predyktor, który służy do detekcji obiektów
        :param state_qualifier: instancja klasy określającej stan obiektów
        :param car_tracker: instancja klasy filtrującej samochody od reszty wykrytych obiektów
        :param lpr: instancja klasy odpowiedzialnej za wykrywanie rejestracji, segmentację i rozpoznawanie liter
        :param frame_skip: liczba klatek, która ma byc pomijana
        :param light_level_th: próg światła, poniżej którego klatka jest ignorowana
        :param prob_th: próg prawdopodobieństwa, poniżej którego obiekt nie jest brany pod uwage
        """
        self.detection_predictor = detection_predictor
        self.state_qualifier = state_qualifier
        self.car_tracker = car_tracker
        self.lpr = lpr
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.light_level_th = light_level_th
        self.prob_th = prob_th
        self.state_dict = None
        self.ids = None
        self.boxes = np.array([])
        self.labels = np.array([])
        self.probabilities = np.array([])
        self.light_on = False
        self.plates = []

    def handle_frame(self, image):
        """
        Metoda obsługująca klatkę. Zwraca kolejno identyfikatory, bounding boxy, etykiety, prawdopodobieństwa,
        słownik tłumaczący indeks samochodu na obecny stan, oraz wartość logiczną mówiącą, czy tablica rejestracyjna
        zaparkowanego samochodu się zgadza
        :param image: obraz w formacie rgb w postaci array'a o kształcie (row, col, 3)
        :return: identyfikatory, bounding boxy, etykiety, float, słownik, true/false
        """

        if self.frame_counter == 0:
            self.light_on = determine_light_on(image, self.light_level_th)
            if self.light_on:
                prediction = self.detection_predictor.predict(image, 15, self.prob_th)
                if prediction[0].size(0) > 0:
                    cars, other, self.ids = self.car_tracker.track('car', prediction)

                    self.state_dict = self.state_qualifier.get_state_dict(cars[0],
                                                                          self.ids,
                                                                          self.car_tracker.prev_values[0],
                                                                          self.car_tracker.prev_values[1])
                    self.boxes = np.vstack((cars[0], other[0]))
                    self.labels = np.concatenate((cars[1], other[1]))
                    self.probabilities = np.concatenate((cars[2], other[2]))
                    self.plates = self._get_plate(cars, self.ids, image)
            else:
                self.ids = np.array([])
                self.boxes = np.array([])
                self.labels = np.array([])
                self.probabilities = np.array([])
                self.state_dict = {}

        self.frame_counter = self.frame_counter + 1
        if self.frame_counter >= self.frame_skip:
            self.frame_counter = 0

        return self.ids, self.boxes, self.labels, self.probabilities, \
               self.state_dict, self.plates, self.light_on

    def _get_plate(self, cars, ids, image):

        """
        Prywatna metoda sprawdzająca dopasowanie odczytu tablicy rejestracyjnej
        :param cars: array z samochodami
        :param ids: identyfikatory samochodów
        :param image: klatka w formacie rgb o kształcie (row, col, 3)
        """
        boxes, _, _ = cars
        parked_ind = -1
        plate = []
        for i in range(0, boxes.shape[0]):
            if self.state_dict[ids[i]] == 'ARRIVED':
                parked_ind = i
                break

        if parked_ind != - 1:
            box = boxes[parked_ind]
            roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            plate = self.lpr.get_license_plate(roi)
        return plate
