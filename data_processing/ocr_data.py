import os

import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from skimage.io import imread
from torch.utils.data import Dataset

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [iaa.Invert(0.5),
     sometimes(iaa.Affine(
         scale={"x": (0.5, 1.0), "y": (0.5, 1.0)},
         translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
         rotate=(-2, 2),
         shear=(0, 0)
     )),
     sometimes(iaa.Dropout(0.1, 0.3)),
     sometimes(iaa.GaussianBlur(1))
     ])


class CharDataset(Dataset):

    def __init__(self, root, label_file, transform=None, char_w=24, char_h=32, multiply=1):
        self.root = root
        self.transform = transform
        self.label_file = label_file
        self.char_w = char_w
        self.char_h = char_h
        self.data, self.class_names, self.class_dict = self._read_data()

        if multiply > 1:
            self._multiply(multiply)

    def __getitem__(self, index):

        images, labels = self.data
        image = images[index]
        label = labels[index]

        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return np.shape(self.data[0])[0]

    def _read_data(self):
        image = imread(self.root + '/charset.png', as_gray=True)
        char_num = np.shape(image)[1] / self.char_w
        image = image.astype(np.float32)
        images = np.hsplit(image, char_num)
        images = np.reshape(images, (-1, 1, self.char_h, self.char_w))
        labels = np.arange(np.shape(images)[0])
        data = images, labels
        label_file_name = f"{self.root}/{self.label_file}"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()
            class_names = class_string.split(',')
            class_dict = {i: class_name for i, class_name in enumerate(class_names)}
        else:
            class_names = []
            class_dict = []
        return data, class_names, class_dict

    def _multiply(self, num):
        ia.seed(np.random.randint(100000, size=1)[0])
        images, labels = self.data
        img, lab = images, labels
        for i in range(0, num):
            images = np.concatenate((images, img), axis=0)
            labels = np.concatenate((labels, lab), axis=0)
        images = self._augment(images)
        self.data = images, labels

    def _augment(self, images):
        images = seq.augment_images(images)
        return images
