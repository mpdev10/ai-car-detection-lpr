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
        self.parked_plate = []

    def handle_frame(self, image):
        """
        Metoda obsługująca klatkę. Zwraca kolejno identyfikatory, bounding boxy, etykiety, prawdopodobieństwa,
        listę potencjalnych odczytów tablicy rejestracyjnej i słownik tłumaczący indeks samochodu na obecny stan
        :param image: obraz w formacie rgb w postaci array'a o kształcie (row, col, 3)
        :return: identyfikatory, bounding boxy, etykiety, prawdopodobieństwa,
        listę potencjalnych odczytów tablicy rejestracyjnej i słownik tłumaczący indeks samochodu na obecny stan
        """
        if self.frame_counter == 0 and determine_light_on(image, self.light_level_th):
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
                self.parked_plate = self._parked_plate_num(cars, self.ids, image)

        self.frame_counter = self.frame_counter + 1
        if self.frame_counter >= self.frame_skip:
            self.frame_counter = 0

        return self.ids, self.boxes, self.labels, self.probabilities, \
               self.parked_plate, self.state_dict

    def _parked_plate_num(self, cars, ids, image):
        """
        Prywatna metoda zwracająca liste potencjalnych odczytów rejestracji
        :param cars: array z samochodami
        :param ids: identyfikatory samochodów
        :param image: klatka w formacie rgb o kształcie (row, col, 3)
        :return: lista stringów będącymi potencjalnymi odczytami rejestracji
        """
        plates = []
        boxes, _, _ = cars
        parked_ind = -1
        for i in range(0, boxes.shape[0]):
            if self.state_dict[ids[i]] == 'ARRIVED':
                parked_ind = i

            if parked_ind != - 1:
                box = boxes[parked_ind]
                roi = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                candidates = self.lpr.perform_ocr(roi)
                if (len(candidates) > 0):
                    plates.append(candidates)
        return plates
