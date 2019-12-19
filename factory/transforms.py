import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from albumentations import (
    OneOf, Compose,
    Flip, ShiftScaleRotate, RandomSizedCrop,
    RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur, GaussianBlur,
    CLAHE, IAASharpen, GaussNoise,
    GridDistortion, ElasticTransform,
    HueSaturationValue, RGBShift,
)


def aug(p=1.0):
    return Compose([
        Flip(p=0.75),
    ], p=p)


def strong_aug(p=1.0):
    return Compose([
        Flip(p=0.75),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        # RandomSizedCrop(min_max_height=(0,0), height=0, width=0, w2h_ratio=0, p=0.3)
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.2),
        # OneOf([
        #     HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.1),
        # GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)


class Albu():
    def __call__(self, image):
        augmentation = aug()
        # augmentation = strong_aug()

        data = {"image": image}
        augmented = augmentation(**data)

        image = augmented["image"]

        return image


class Albu_test():
    def __call__(self, image):
        augmentation = strong_aug()

        data = {"image": image}
        augmented = augmentation(**data)

        image = augmented["image"]

        return image


class VFlip:
    def __call__(self, image):
        return np.flip(image, axis=0).copy()
        # return image[::-1]


class HFlip:
    def __call__(self, image):
        return np.flip(image, axis=1).copy()
        # return image[:,::-1]


if __name__ == "__main__":
    aug = Albu_test()

    img = cv2.imread('000000_01.jpg', 1)
    img = cv2.resize(img, (288, 224))

    for i in range(100):
        out_img = aug(img)

        cv2.imshow('img', out_img)
        cv2.waitKey()
