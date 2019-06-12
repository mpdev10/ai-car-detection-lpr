import cv2
import numpy as np
from skimage import measure
from skimage.filters import threshold_isodata
from skimage.measure import regionprops


class LicensePlateDetector:
    def __init__(self, plate_dimensions):
        min_height, max_height, min_width, max_width = plate_dimensions
        self.min_w = min_width
        self.max_w = max_width
        self.min_h = min_height
        self.max_h = max_height

    def find_candidates(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold_value = threshold_isodata(image)
        binary_car_image = image > threshold_value
        label_image = measure.label(binary_car_image)
        candidates = []
        for region in regionprops(label_image):
            if region.area < 50:
                continue
            min_row, min_col, max_row, max_col = region.bbox
            bbox = min_col, min_row, max_col, max_row
            region_height = max_row - min_row
            region_width = max_col - min_col
            if self.min_h <= region_height <= self.max_h and self.min_w <= region_width <= self.max_w \
                    and region_width >= region_height * 3 and min_row >= (image.shape[0] / 2):
                candidates.append((np.invert(binary_car_image[min_row:max_row, min_col:max_col]), bbox))
        return candidates
