import cv2
import imutils
import numpy as np
from skimage import measure
from skimage.measure import regionprops


class LicensePlateDetector:
    """
    Klasa odpowiedzialna za detekcję tablicy rejestracyjnej na obrazie
    """

    def __init__(self, plate_dimensions):
        """
        :param plate_dimensions: wymiary tablicy rejestracyjnej w postaci krotki (minh, maxh, minw, maxw)
        """
        min_height, max_height, min_width, max_width = plate_dimensions
        self.min_w = min_width
        self.max_w = max_width
        self.min_h = min_height
        self.max_h = max_height

    def find_candidates(self, image):
        """
        Metoda zwraca potencjalne tablice rejestracyjne
        :param image: obraz w postaci array'a o kształcie (row, col, 3)
        :return: lista krotek złożonych z dwuwymiarowych arrayów i koordynatów wykrytych kandydatów
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary_car_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
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
                candidates.append((self._warp_image(image[min_row:max_row, min_col:max_col]), bbox))
        return candidates

    @staticmethod
    def _warp_image(image):
        """
        Metoda dokonuje przekształcenie obrazu, aby był on zbliżony do prostokąta
        :param image: obraz
        :return: przekształcony obraz
        """
        gray = cv2.cvtColor(image[:, 0:int(image.shape[1])], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.dilate(thresh, (3, 3), iterations=1)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        hull = cv2.convexHull(c, False)

        mask = np.full(shape=gray.shape, fill_value=0, dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, thickness=2)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if approx.shape[0] >= 4:
            left_top = min(approx, key=lambda x: x[0][0] + x[0][1])
            right_top = min(approx, key=lambda x: (image.shape[1] - x[0][0]) + x[0][1])
            left_bottom = min(approx, key=lambda x: (image.shape[0] - x[0][1]) + x[0][0])
            right_bottom = min(approx, key=lambda x: (image.shape[1] + image.shape[0]) - (x[0][0] + x[0][1]))

            pts1 = np.float32([left_top, right_top, left_bottom, right_bottom])
            pts2 = np.float32(
                [[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            image = dst
        return image
