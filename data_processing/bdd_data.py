import json

import cv2
from pandas.io.json import json_normalize
from skimage import io


class BDDFormatDataset:

    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.data, self.class_names, self.class_dict = self._read_data()
        self.min_image_num = -1
        self.ids = [info['image_id'] for info in self.data]

    def _read_data(self):
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations.json"

        with open(annotation_file) as f:
            annotations = json.load(f)
        annotations = json_normalize(annotations)
        normalized_list = annotations.labels.apply(lambda labels: list() if not labels else list(labels))
        categories = set()
        for labels in normalized_list:
            for label in labels:
                categories.add(label['category'])
        class_names = sorted(list(categories))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []

        for image, group in annotations.groupby('name'):
            boxes = []
            labels = []

            for labels_l in group.labels:
                if labels_l:
                    for label in labels_l:
                        box = label['box2d']
                        boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
                        labels.append(class_dict[label['category']])
            data.append({
                'image_id': image,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def _read_image(self, image_id):
        image = io.imread(image_id)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = image_info['boxes']
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels
