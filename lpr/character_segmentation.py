import numpy as np
from cv2 import cv2
from skimage import measure


class CharSeg:
    """
    Klasa odpowiedzialna za segmentację znaków na obrazie
    """

    def __init__(self, character_dimensions, padding=0):
        """
        :param character_dimensions: wymiary znaku w postaci array'a [minh, maxh, minw, maxw]
        :param padding: wielkość marginesu dookoła wykrytego znaku w pikselach
        """
        self.padding = padding
        self.character_dimensions = character_dimensions

    def segment_image(self, image):
        """
        Metoda segmentuje obraz na litery
        :param image:
        :return: lista krotek o postaci (img, [ymin, xmin, ymax, xmax]), gdzie img to dwuwymiarowy array
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (1, 1), 0)

        thresh = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        fixed_boxes = self._segment_plate(thresh, 8)
        return self._prepare_boxes(thresh, fixed_boxes)

    @staticmethod
    def _segment_plate(image, segment_number):
        """
        Metoda dokonuje segmentacji na regiony
        :param image: obraz z tablicą rejestracyjną
        :param segment_number: liczba segmentów, na które ma być podzielona tablica
        :return: lista krotek (y0, x0, y1, x1)
        """
        i_w = image.shape[1]
        i_h = image.shape[0]
        min_h = int(i_h - (6 / 10) * i_h)
        max_h = int(i_h - (4 / 10) * i_h)
        offset = 0
        seg_w = int(round(i_w / segment_number, 0))
        boxes = []
        for i in range(0, segment_number):
            pixel_min = 1

            if offset + seg_w * i + seg_w < i_w - 1:
                pixel_min = np.min(image[min_h:max_h, offset + seg_w * i + seg_w])
            while offset + seg_w * i + seg_w < i_w - 1 and pixel_min <= 0:
                offset = offset + 1
                pixel_min = np.min(image[min_h:max_h, offset + seg_w * i + seg_w])
            if offset + seg_w * i + seg_w < i_w - 1:
                boxes.append((0, offset + seg_w * i, i_h - 1, offset + seg_w * i + seg_w))
        return boxes

    def _prepare_boxes(self, image, boxes):
        """
        Metoda przygotowuje regiony, dodając do nich margines (padding)
        :param regions: wynik metody regionprops
        :return: lista krotek (image, bbox)
        """
        segments = []
        for bbox in boxes:
            if self._is_valid_region(bbox):
                y0, x0, y1, x1 = bbox
                y0 = y0 - self.padding if y0 - self.padding > 0 else 0
                y1 = y1 + self.padding if y1 + self.padding < image.shape[0] else image.shape[0] - 1
                x0 = x0 - self.padding if x0 - self.padding > 0 else 0
                x1 = x1 + self.padding if x1 + self.padding < image.shape[1] else image.shape[1] - 1
                roi = image[y0:y1, x0:x1]
                labels = measure.label(roi, background=255)
                props = list(filter(lambda x: self._is_valid_region(x.bbox), measure.regionprops(labels)))
                by0, bx0, by1, bx1 = (0, 0, y1 - y0, x1 - x0)
                if props.__len__() > 0:
                    biggest = max(props, key=lambda x: x.filled_area)
                    by0, bx0, by1, bx1 = biggest.bbox

                segment = (roi[by0:by1, bx0:bx1], (y0 + by0, x0 + bx0, y0 + by1, x0 + bx1))
                segments.append(segment)
                segments.sort(key=lambda x: x[1][1])
        return segments

    def _is_valid_region(self, bbox):
        """
        Metoda sprawdza, czy region kwalifikuję się jako tablica rejestracyjna
        :param region: wynik metody regionprops z skimage
        :return: True lub False
        """
        y0, x0, y1, x1 = bbox
        region_height = y1 - y0
        region_width = x1 - x0
        min_height, max_height, min_width, max_width = self.character_dimensions
        return min_height <= region_height <= max_height and min_width <= region_width <= max_width \
               and region_width < region_height
