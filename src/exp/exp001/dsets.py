from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

import glob
import cv2
import numpy as np
import os

import pydicom

def get_train_transforms(img_size):
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #ShiftScaleRotate(p=1.0),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        # CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(img_size):
    return Compose([
        # CenterCrop(img_size, img_size, p=1.),
        #Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_inference_transforms(img_size):
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=1.0),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

class BrainTumor2dSimpleDataset(Dataset):
    def __init__(self, df, img_size, transforms=None, output_label=True):
        self.paths = df["BraTS21ID"].values
        self.output_label = output_label
        if self.output_label:
            self.targets = df["MGMT_value"].values
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = f'./input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{str(_id).zfill(5)}/'
        channels = []
        for t in ("FLAIR", "T1w", "TqwCE"):
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, '*')),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )

            x = len(t_paths)
            if x < 10:
                r = range(x)
            else:
                d = x // 10
                r = range(d, x-d, d)
            channel = []
            for i in r:
                channel.append(cv2.resize(load_dicom(t_paths[i]), (self.img_size, self.img_size)))
            channel = np.mean(channel, axis=0)
            channels.append(channel)

        if self.transforms:
            img = self.transforms(image=channels)['image']

        if self.output_label:
            target = torch.tensor(self.targets[index], dtype=torch.float)
            return img, target
        else:
            return img

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data