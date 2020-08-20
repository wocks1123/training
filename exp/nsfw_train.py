
import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from core.object_detection import training, evaluation, inference

CONFIG_PATH = "config/train_nsfw.json"


def train_nsfw():
    transforms = {
        "TRAIN": A.Compose([
            A.Resize(300, 300),
            A.HorizontalFlip(p=0.5),
            A.MotionBlur(p=0.5),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}),
        "TEST": A.Compose([
            A.Resize(300, 300),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    }

    training(
        config_path=CONFIG_PATH,
        transforms=transforms
    )


def eval_nsfw():
    checkpoint_path = "data/weight/2020-08-20_15_05_07_mobilenetv2_q_ssd_pascalvoc_v2/epoch20_checkpoint.pth.tar"  # mobilenet

    evaluation(
        config_path=CONFIG_PATH,
        checkpoint_path=checkpoint_path,
        transforms={
            "TEST": A.Compose([
                A.Resize(300, 300),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        }
    )


def inference_img():
    checkpoint_path = "data/weight/2020-08-20_15_05_07_mobilenetv2_q_ssd_pascalvoc_v2/epoch20_checkpoint.pth.tar"  # mobilenet

    img_path = "/DB/VOC2007_test100/JPEGImages/000005.jpg"
    inference(
        config_path=CONFIG_PATH,
        checkpoint_path=checkpoint_path,
        img_path=img_path,
        transforms={
            "TEST": A.Compose([
                A.Resize(300, 300),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        }
    )

