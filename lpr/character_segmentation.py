from skimage import measure
from skimage.measure import regionprops


class CharSeg:
    def __init__(self, character_dimensions, padding=2):
        self.padding = padding
        self.character_dimensions = character_dimensions

    def segment_image(self, image):
        labelled_image = measure.label(image)
        segments = []
        for region in regionprops(labelled_image):
            if self._is_valid_region(region):
                y0, x0, y1, x1 = region.bbox
                y0 = y0 - self.padding if y0 - self.padding > 0 else 0
                y1 = y1 + self.padding if y1 + self.padding < image.shape[0] else image.shape[0] - 1
                x0 = x0 - self.padding if x0 - self.padding > 0 else 0
                x1 = x1 + self.padding if x1 + self.padding < image.shape[1] else image.shape[1] - 1
                segment = image[y0:y1, x0:x1], region.bbox
                segments.append(segment)
        return segments

    def _is_valid_region(self, region):
        y0, x0, y1, x1 = region.bbox
        region_height = y1 - y0
        region_width = x1 - x0
        min_height, max_height, min_width, max_width = self.character_dimensions
        return min_height <= region_height <= max_height and min_width <= region_width <= max_width
