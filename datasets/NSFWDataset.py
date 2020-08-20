
import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)


import torch
from torch.utils.data import Dataset

from PIL import Image

from utils.utils import transform


def yolo2pascalvoc(width, height, x, y, w, h):
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    return [xmin, ymin, xmax, ymax]


NSFW_LABELS = ("man", "woman")


class NSFW_SSDDateset(Dataset):
    def __init__(self, anno_txt_path, split):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        with open(anno_txt_path, "r") as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        curr_img_path = self.lines[idx]
        curr_img_path = curr_img_path.rstrip()

        img = Image.open(curr_img_path)
        img = img.convert('RGB')

        annotation_file = curr_img_path[:-3] + "txt"
        boxes = []
        labels = []
        with open(annotation_file, "r") as f:
            anno_lines = f.readlines()
            for l in anno_lines:
                c, x, y, w, h = l.split(" ")
                boxes.append(yolo2pascalvoc(img.width, img.height, float(x), float(y), float(w), float(h)))
                labels.append(int(c) + 1)

        difficulties = torch.ByteTensor([0 for i in range(len(labels))])
        boxes = torch.FloatTensor(boxes)
        labels = torch.tensor(labels, dtype=torch.int64)


        image, boxes, labels, difficulties = transform(img, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.lines)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties # tensor (N, 3, 300, 300), 3 lists of N tensors each