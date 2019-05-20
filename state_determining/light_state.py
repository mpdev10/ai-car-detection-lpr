import cv2

import numpy as np


def light_level(image):
    return np.average(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def determine_light_on(image, threshold):
    return light_level(image) >= threshold
