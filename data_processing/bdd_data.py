import json

import cv2
import numpy as np
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
            labels = np.empty
            boxes = None
            for labels_l in group.labels:
                if labels_l:
                    for label in labels_l:
                        box = label['box2d']
                        curr_box = np.array([[box['x1'], box['y1'], box['x2'], box['y2']]]).astype(np.float32)
                        if boxes is None:
                            boxes = curr_box
                        else:
                            boxes = np.vstack((boxes, curr_box))
                            labels = np.append(labels, [class_dict[label['category']]])
            if boxes is not None:
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

    def get_annotation(self, index):
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)
