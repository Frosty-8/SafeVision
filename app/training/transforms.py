from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms():

    return A.Compose(

        [

            A.Resize(
                640,
                640,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            A.GaussNoise(
                p=0.2,
            ),

            A.MotionBlur(
                blur_limit=5,
                p=0.2,
            ),

            A.Normalize(),

            ToTensorV2(),

        ],

        bbox_params=A.BboxParams(

            format="pascal_voc",

            label_fields=[
                "labels",
            ],

        ),

    )

def build_validation_transforms():

    return A.Compose(

        [

            A.Resize(
                640,
                640,
            ),

            A.Normalize(),

            ToTensorV2(),

        ],

        bbox_params=A.BboxParams(

            format="pascal_voc",

            label_fields=[
                "labels",
            ],

        ),

    )

