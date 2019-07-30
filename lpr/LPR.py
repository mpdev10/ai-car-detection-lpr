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

    def check_license_plate(self, image, plate_str):
        candidateList = self._perform_ocr(image)
        max_matching_chars = 0
        for candidate in candidateList:
            matching_chars = 0
            i = 0
            j = 0
            while j < len(plate_str):
                pchar = plate_str[j]
                if i >= len(candidate):
                    break
                if candidate[i][pchar] < self.max_char_place:
                    matching_chars = matching_chars + 1
                    j = j + 1
                i = i + 1
            if matching_chars > max_matching_chars:
                max_matching_chars = matching_chars

        return max_matching_chars / len(plate_str)

    def _perform_ocr(self, image):
        """
        Metoda zwraca odczyty znaków z potencjalnych wykrytych tablic rejestracyjnych
        :param image: obraz w postaci array'a o kształcie (row, col, 3)
        :return: lista list słowników, gdzie wartość slownik[char] oznacza miejsce char pod względem prawdopodobieństwa
        """
        candidates = self.plate_detector.find_candidates(image)
        possible_platenums = []
        for candidate in candidates:
            plate_img, _ = candidate
            segments = self.char_seg.segment_image(plate_img)
            if len(segments) > 6:
                chars = []
                for i in range(len(segments)):
                    segment = segments[i]
                    tensor_output = self._classify(segment[0])
                    chars.append(self._char_map(tensor_output))
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

    def _char_map(self, tensor_output):
        """
        Zwraca mapę, która przypisuje literom ich miejsce pod względem prawdopodobieństwa
        :param tensor_output: tensor.Torch - wynik metody _classify
        :return: mapa/słownik
        """
        ret = np.flip(np.argsort(np.asarray(tensor_output.detach())).flatten())
        ret = [self._get_character(x) for x in ret]
        charMap = {}
        for i in range(0, len(ret)):
            charMap[ret[i]] = i
        return charMap

    def _get_character(self, num):
        """
        Zwraca znak odpowiadający podanemu numerowi klasy
        :param num: indeks klasy
        :return: znak/char
        """
        ret = self.dataset.class_dict[num]
        return ret
