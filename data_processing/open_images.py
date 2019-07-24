import pathlib

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class OpenImagesDataset(Dataset):
    """
    Klasa obsługująca dataset w formacie OpenImages (https://storage.googleapis.com/openimages/web/index.html)
    """

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        """
        :param root: ścieżka do katalogu z plikiem sub-[typ]-dataset.json
        :param transform: klasa aplikująca transformacje na każdy obraz w datasecie
        :param target_transform: klasa aplikująca transformację na bounding box'y i etykiety w datasecie
        :param dataset_type: typ datasetu; train lub test
        :param balance_data: jeżeli True to wyrównuje liczbe obrazów w datasecie, aby było równo dla każdej z etykiet
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        """
        Metoda zwracająca id, obraz, bounding box'y i etykiety dla danego indeksu
        :param index: indeks obrazu
        :return: krotka w postaci (id, image, boxes, labels)
        """
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

    def _read_data(self):
        """
        Metoda parsująca plik csv z etykietami
        """
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            labels = np.array([class_dict[name] for name in group["ClassName"]])
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        """
        Metoda zwraca wielkość datasetu
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

    def _read_image(self, image_id):
        """
        Metoda zwracajacą obraz o podanym identyfikatorze
        :param image_id: id obrazu
        :return: array o kształcie (row, col, 3)
        """
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

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
