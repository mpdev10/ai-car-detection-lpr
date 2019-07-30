import os

import imgaug as ia
import numpy as np
from cv2 import cv2
from imgaug import augmenters as iaa
from skimage.io import imread
from torch.utils.data import Dataset

sometimes = lambda aug: iaa.Sometimes(0.95, aug)

seq = iaa.Sequential(
    [
        iaa.Affine(
            scale={"x": (0.3, 1.4), "y": (0.3, 1.4)},
            rotate=(-20, 20)
        ),
        sometimes([
            iaa.Invert(0.5),
            iaa.Dropout((0, 0.4)),
            iaa.PerspectiveTransform(scale=(0.01, 0.02)),
            iaa.GaussianBlur((0, 1.5))])

    ])


class CharDataset(Dataset):
    """
    Klasa obsługująca dataset dla znaków
    """

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
        image = imread(self.root + '/charset.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        char_num = np.shape(image)[1] / self.char_w
        images = np.hsplit(image, char_num)
        images = np.reshape(images, (-1, self.char_h, self.char_w))
        cv2.waitKey(0)

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

        images, labels = self.data
        img, lab = images, labels
        for i in range(0, num):
            images = np.concatenate((images, img), axis=0)
            labels = np.concatenate((labels, lab), axis=0)
        images = self._augment(images)
        self.data = images, labels

    def _augment(self, images):
        ia.seed(np.random.randint(9999999, size=1)[0])
        images = seq.augment_images(images)
        images = np.reshape(images, (-1, 1, self.char_h, self.char_w))
        return images
