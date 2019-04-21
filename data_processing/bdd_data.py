class BDDFormatDataset:

    def __init__(self, root, transform=None, target_transform=None,
                 dataset_type='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

