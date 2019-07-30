from cv2 import cv2
from skimage import measure
from skimage.measure import regionprops


class CharSeg:
    """
    Klasa odpowiedzialna za segmentację znaków na obrazie
    """

    def __init__(self, character_dimensions, padding=2):
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
        gray = cv2.bilateralFilter(gray, 20, 17, 17)
        canny = cv2.Canny(gray, 50, 200)

        labelled_image = measure.label(canny)
        props = regionprops(labelled_image)
        fixed_boxes = self._get_proper_boxes(props)
        return self._prepare_boxes(gray, fixed_boxes)

    def _get_proper_boxes(self, props):
        global prev_region
        props.sort(key=lambda x: x.centroid[1])
        regions = []
        boxes = []
        w_sum = 0
        for region in props:
            y0, x0, y1, x1 = region.bbox
            if self._is_valid_region(region.bbox):
                w_sum = w_sum + (x1 - x0)
                regions.append(region)
        avg_w = 0 if len(regions) == 0 else w_sum / len(regions)
        prev_x = -1
        for i in range(0, len(regions)):
            region = regions[i]
            y0, x0, y1, x1 = region.bbox
            if prev_x != -1:
                dist = region.centroid[1] - prev_x
                if dist > avg_w * 2:
                    py0, px0, py1, px1 = prev_region.bbox
                    fx0 = (x0 + px0) / 2
                    fy0 = (y0 + py0) / 2
                    fx1 = (x1 + px1) / 2
                    fy1 = (y1 + py1) / 2
                    new_bbox = int(fy0), int(fx0), int(fy1), int(fx1)
                    boxes.append(new_bbox)
                if dist > 5:
                    boxes.append(region.bbox)
            else:
                boxes.append(region.bbox)
            prev_x = region.centroid[1]
            prev_region = region

        return boxes

    def _prepare_boxes(self, image, boxes):
        """
        Metoda przygotowuje regiony, dodając do nich margines (padding)
        :param regions: wynik metody regionprops
        :return: krotka (image, bbox)
        """
        segments = []
        for bbox in boxes:
            if self._is_valid_region(bbox):
                y0, x0, y1, x1 = bbox
                y0 = y0 - self.padding if y0 - self.padding > 0 else 0
                y1 = y1 + self.padding if y1 + self.padding < image.shape[0] else image.shape[0] - 1
                x0 = x0 - self.padding if x0 - self.padding > 0 else 0
                x1 = x1 + self.padding if x1 + self.padding < image.shape[1] else image.shape[1] - 1
                segment = image[y0:y1, x0:x1], bbox
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
        return min_height <= region_height <= max_height and min_width <= region_width <= max_width
