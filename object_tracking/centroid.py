import numpy as np


def compute_centroids(boxes):
    """
    Metoda wylicza środki prostokątów podanych jako argument metody
    :param boxes: array o kształcie (n, 4), gdzie n to liczba prostokątów, xmin - boxes[:, 0], xmax - boxes[:, 2], itd.
    :return: array o kształcie (n, 2), gdzie n to liczba prostokątów, array[:, 0] to x, a array[:, 1] to y
    """
    centroids = np.vstack(((boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2))
    return centroids.transpose()


def euclidean_distance(p_from, p_to):
    """
    Metoda wyliczająca odległość pomiędzy punktem p_from, a punktem p_from
    :param p_from: punkt w postaci jednowymiarowego array'a o rozmiarze 2, gdzie pierwszy indeks oznacza x a drugi y
    :param p_to: punkt w postaci jednowymiarowego array'a o rozmiarze 2, gdzie pierwszy indeks oznacza x a drugi y
    :return: wartość liczbowa, oznaczająca odległość między punktami
    """
    return np.sum(np.abs(p_from - p_to))


def intersect_exists(box1, box2):
    """
    Metoda sprawdza, czy dwa prostokąty mają część wspólną w przestrzeni euklidesowej
    :param box1: array o rozmiarze 4, gdzie xmin - 0, ymin - 1, xmax - 2, ymax - 3
    :param box2: array o rozmiarze 4, gdzie xmin - 0, ymin - 1, xmax - 2, ymax - 3
    :return: True - jeżeli mają część wspólną, lub False - jeżeli nie
    """
    return box1[2] >= box2[0] and box1[0] <= box2[2] and box1[3] >= box2[1] and box1[1] <= box2[3]
