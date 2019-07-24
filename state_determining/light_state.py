import cv2

import numpy as np


def light_level(image):
    """
    Metoda zwraca średni poziom jasności obrazu
    :param image: obraz w postaci array'a o kształcie (row, col, 3)
    :return: wartość liczbowa określająca średni poziom jasności obrazu
    """
    return np.average(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def determine_light_on(image, threshold):
    """
    Metoda sprawdza, czy poziom średni poziom jasności jest wyższy lub równy progowi podanemu jako parametr
    :param image: obraz w postaci array'a o kształcie (row, col, 3)
    :param threshold: wartość liczbowa od 0 do 255
    :return: True lub False
    """
    return light_level(image) >= threshold
