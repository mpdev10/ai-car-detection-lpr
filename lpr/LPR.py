import cv2
import numpy as np
import torch

from lpr.character_segmentation import CharSeg
from lpr.cnn import CNN
from lpr.license_plate_detector import LicensePlateDetector


class LPR:
    """
    Klasa agregująca wszystkie podmoduły związane z detekcją i odczytem tablic rejestracyjnych
    """

    def __init__(self, char_seg: CharSeg, plate_detector: LicensePlateDetector, model: CNN,
                 dataset, input_dim=(24, 32), max_char_place=4):
        """
        :param char_seg: instancja klasy CharSeg do segmentacji obrazu na znaki
        :param plate_detector: instancja klasy LicensePlateDetector do wykrywania rejestracji
        :param model: model sieci neuronowej do klasyfikacji znaków
        :param dataset: dataset posiadający słownik, który tłumaczy liczby na nazwy klas
        :param input_dim: krotka w postaci (w, h), gdzie w to szerokość, a h wysokość litery; domyślnie (24, 32)
        :param max_char_place: maksymalne miejsce, które może zajmować znak, aby być uznany za pasujący
        """
        self.char_seg = char_seg
        self.plate_detector = plate_detector
        self.model = model
        self.dataset = dataset
        self.char_w = input_dim[0]
        self.char_h = input_dim[1]
        self.max_char_place = max_char_place

    def get_license_plate(self, image):
        candidateList = self._perform_ocr(image)
        return candidateList

    def _perform_ocr(self, image):
        """
        Metoda zwraca odczyty znaków z potencjalnych wykrytych tablic rejestracyjnych
        :param image: obraz w postaci array'a o kształcie (row, col, 3)
        :return: lista stringów
        """
        candidates = self.plate_detector.find_candidates(image)
        possible_platenums = []
        for candidate in candidates:
            plate_img, _ = candidate
            segments = self.char_seg.segment_image(plate_img)
            if 9 > len(segments) > 6:
                chars = ""
                for i in range(len(segments)):
                    segment = segments[i]
                    tensor_output = self._classify(segment[0])
                    chars = chars + self._get_character(np.argmax(tensor_output.detach().numpy()))
                possible_platenums.append(chars)
        return possible_platenums

    def _classify(self, image):
        """
        Prywatna metoda dokonująca klasyfikacji obrazu
        :param image: obraz w postaci dwuwymiarowego array'a
        :return: tensor.Torch, gdzie wartosć z tensora[i] oznacza prawdopodobieństwo, że znak na obrazie jest klasy i
        """
        reshaped = cv2.resize(image, (self.char_w, self.char_h))
        reshaped = np.invert(reshaped)
        return self.model(torch.Tensor(reshaped))

    def _get_character(self, num):
        """
        Zwraca znak odpowiadający podanemu numerowi klasy
        :param num: indeks klasy
        :return: znak/char
        """
        ret = self.dataset.class_dict[num]
        return ret
