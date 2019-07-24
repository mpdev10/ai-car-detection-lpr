import json
import os

import cv2
import numpy as np
from pandas.io.json import json_normalize
from skimage import io
from torch.utils.data import Dataset


class BDDFormatDataset(Dataset):
    """
    Klasa obsługująca dataset utworzony w narzędziu Scalabel (https://www.scalabel.ai/)
    """

    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train', label_file=None, balance_data=False):
        """
        :param root: ścieżka do katalogu z plikiem sub-[typ]-dataset.json
        :param transform: klasa aplikująca transformacje na każdy obraz w datasecie
        :param target_transform: klasa aplikująca transformację na bounding box'y i etykiety w datasecie
        :param dataset_type: typ datasetu; train lub test
        :param label_file: plik, w którym wymienione są nazwy etykiet w odpowiedniej kolejności, oddzielone przecinkiem
        :param balance_data: jeżeli True to wyrównuje liczbe obrazów w datasecie, aby było równo dla każdej z etykiet
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.label_file = label_file
        self.data, self.class_names, self.class_dict = self._read_data()
        self.min_image_num = -1
        self.balance_data = balance_data
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]
        self.class_stat = None

    def _read_data(self):
        """
        Metoda parsująca plik json z etykietami
        """
        global class_names
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations.json"

        with open(annotation_file) as f:
            annotations = json.load(f)
        annotations = json_normalize(annotations)
        if self.label_file is not None:
            label_file_name = f"{self.root}/{self.label_file}"
            if os.path.isfile(label_file_name):
                class_string = ""
                with open(label_file_name, 'r') as infile:
                    for line in infile:
                        class_string += line.rstrip()
                class_names = class_string.split(',')
                class_names.insert(0, 'BACKGROUND')
        else:
            normalized_list = annotations.labels.apply(lambda labels: list() if not labels else list(labels))
            categories = set()
            for labels in normalized_list:
                for label in labels:
                    categories.add(label['category'])
            class_names = ['BACKGROUND'] + sorted(list(categories))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []

        for image, group in annotations.groupby('name'):
            labels = None
            boxes = None
            for labels_l in group.labels:
                if labels_l:
                    for label in labels_l:
                        box = label['box2d']
                        curr_box = np.array([[box['x1'], box['y1'], box['x2'], box['y2']]]).astype(np.float32)
                        if boxes is None:
                            boxes = curr_box
                            labels = np.array([class_dict[label['category']]])
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
        """
        Metoda zwracajacą obraz o podanym identyfikatorze
        :param image_id: id obrazu
        :return: array o kształcie (row, col, 3)
        """
        image = io.imread(image_id)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image(self, index):
        """
        Metoda zwracająca obraz na podanym indeksie
        :param index: indeks obrazu
        :return: array o kształcie (row, col, 3)
        """
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _getitem(self, index):
        """
        Metoda zwracająca id, obraz, bounding box'y i etykiety dla danego indeksu
        :param index: indeks obrazu
        :return: krotka w postaci (id, image, boxes, labels)
        """
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = image_info['boxes']
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        """
        Metoda zwracająca obraz, bounding box'y i etykiety dla danego indeksu
        :param index: indeks obrazu
        :return: krotka array'ów o kształtach: image: (row, col, 3), boxes (n, 4), labels (n, 1)
        """
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """
        Metoda zwraca identyfikator i etykiety dla danego indeksu
        :param index:
        :return: id, (boxes, labels, is_difficult); kształty array'ów: boxes: (n, 4), labels (n, 1)
        """
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def __len__(self):
        """
        :return: Wielkość datasetu
        """
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

    def _balance_data(self):
        """
        Metoda balansująca dataset
        """
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data
