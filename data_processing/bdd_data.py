import json

from pandas.io.json import json_normalize


class BDDFormatDataset:

    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self._read_data()

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
