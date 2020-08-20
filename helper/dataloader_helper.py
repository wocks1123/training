
import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)


import torch

from datasets.CIFAR10Dataset import CIFAR10Dataset, CIFAR10_CLASSES
from datasets.NSFWDataset import NSFW_SSDDateset, NSFW_LABELS
from datasets.PascalVOCDataset import PascalVOCDataset, PASCAL_CLASSES
from datasets.PascalVOCDataset_v2 import PascalVOCDataset_v2, PASCAL_CLASSES


def get_cifar10(data_dir, transforms):
    dataset = CIFAR10Dataset(
        data_dir=data_dir,
        transform=transforms
    )

    return dataset, CIFAR10_CLASSES


def get_pascalvoc(data_folder, split, transforms=None):
    dataset = PascalVOCDataset(
        data_folder,
        split,
        keep_difficult=True
    )

    return dataset, PASCAL_CLASSES


def get_pascalvoc_v2(data_folder, split, transforms=None):
    dataset = PascalVOCDataset_v2(
        data_folder,
        split,
        True,
        transforms
    )

    return dataset, PASCAL_CLASSES


def get_nsfw(anno_txt_path, split):
    dataset = NSFW_SSDDateset(
        anno_txt_path,
        split
    )

    return dataset, NSFW_LABELS


def load_dataset(dataset_name, data_dir, split, options, transforms, batch_size):
    num_workers = options["num_workers"]

    if dataset_name == "cifar10":
        dataset, class_list = get_cifar10(data_dir=data_dir, transforms=transforms)
    elif dataset_name == "pascalvoc":
        dataset, class_list = get_pascalvoc(data_folder=data_dir, split=split)
    elif dataset_name == "pascalvoc_v2":
        dataset, class_list = get_pascalvoc_v2(data_folder=data_dir, split=split, transforms=transforms)
    elif dataset_name == "nsfw":
        dataset, class_list = get_nsfw(anno_txt_path=data_dir, split=split)
    else:
        raise Exception("Wrong Dataset Name... Check config.json")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader, class_list
