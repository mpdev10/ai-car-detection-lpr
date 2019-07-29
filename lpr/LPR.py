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
                 dataset, input_dim=(24, 32)):
        """
        :param char_seg: instancja klasy CharSeg do segmentacji obrazu na znaki
        :param plate_detector: instancja klasy LicensePlateDetector do wykrywania rejestracji
        :param model: model sieci neuronowej do klasyfikacji znaków
        :param dataset: dataset posiadający słownik, który tłumaczy liczby na nazwy klas
        :param input_dim: krotka w postaci (w, h), gdzie w to szerokość, a h wysokość litery; domyślnie (24, 32)
        """
        self.char_seg = char_seg
        self.plate_detector = plate_detector
        self.model = model
        self.dataset = dataset
        self.char_w = input_dim[0]
        self.char_h = input_dim[1]

    def perform_ocr(self, image):
        """
        Metoda zwraca odczyty znaków z potencjalnych wykrytych tablic rejestracyjnych
        :param image: obraz w postaci array'a o kształcie (row, col, 3)
        :return: jednowymiarowy array stringów
        """
        candidates = self.plate_detector.find_candidates(image)
        possible_platenums = []
        for candidate in candidates:
            plate_img, _ = candidate
            segments = self.char_seg.segment_image(plate_img)
            if len(segments) > 3:
                chars = []
                column_list = []
                txt = ''
                for i in range(len(segments)):
                    segment = segments[i]
                    y0, x0, y1, x1 = segment[1]
                    reshaped = cv2.resize(segment[0], (self.char_w, self.char_h))

                    ret = np.asarray(self.model(torch.Tensor(reshaped)).detach())
                    cv2.imshow('a', reshaped)
                    chars.append(self.dataset.class_dict[np.argmax(ret)])
                    column_list.append(x0)
                for i in np.argsort(column_list):
                    txt = txt + chars[i]
                possible_platenums.append(txt)
        return possible_platenums
