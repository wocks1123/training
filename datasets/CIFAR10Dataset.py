
import torch
from torch.utils.data import Dataset
import cv2

import os


CIFAR10_CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'deer',
    'dog',
    'cat',
    'frog',
    'horse',
    'truck',
    'ship'
]


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.datas = []
        self.transform = transform
        for class_name in CIFAR10_CLASSES:
            class_dir = f"{data_dir}/{class_name}"
            files = os.listdir(class_dir)
            for file_name in files:
                file_path = os.path.join(class_dir, file_name)
                self.datas.append({"path": file_path, "label": CIFAR10_CLASSES.index(class_name)})

    def __getitem__(self, item):
        # image = Image.open(f"{self.datas[item]['path']}", mode='r')
        # image = image.convert('RGB')
        image = cv2.imread(f"{self.datas[item]['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.datas[item]['label']

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        images = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            labels.append(b[1])

        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        return images, labels