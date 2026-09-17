from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import math
import logging
import torch
import torch.nn as nn

from pydantic import BaseModel, ConfigDict

from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
)

def percentile(image:np.ndarray, normalization:str) -> np.ndarray :
    if normalization == "percentile":
        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)
        image_norm = (image - p1) / (p99 - p1)

    if normalization == "zscore":
        mu_pre = np.mean(image)
        sigma_pre = np.std(image)
        image_norm = (image - mu_pre) / (sigma_pre)

    return image_norm


class dataLoading():
    def __init__(
        self, 
        folder_path_0,
        folder_path_1, 
        target_size = (200,200)
    ):
        super().__init__()

        self.folder_path_0 = folder_path_0
        self.folder_path_1 = folder_path_1
        self.target_size = target_size
    

    def _load_data(self):
        self.image = []
        self.label = []
        
        for file_name in os.listdir(self.folder_path_0):
            file_path = os.path.join(self.folder_path_0, file_name)
            image = Image.open(file_path).convert("L")
            if self.target_size:
                image = image.resize(self.target_size)
            image = np.array(image, dtype=np.uint8)
            self.image.append(image)
            self.label.append(0)

        for file_name in os.listdir(self.folder_path_1):
            file_path = os.path.join(self.folder_path_1, file_name)
            image = Image.open(file_path).convert("L")
            if self.target_size:
                image = image.resize(self.target_size)
            image = np.array(image, dtype=np.uint8)
            self.image.append(image)
            self.label.append(1)

        return self.image, self.label


class LoadingPreProcessing():
    def __init__(self, 
        image : list, 
        label : list, 
        normalization : str
    ):
        super().__init__()
        self.normalization = normalization
        self.image = image
        self.label = label
        
    def __len__(self):
        return len(self.image)

    def __getitem__(
        self, index: int
        ) -> (tuple[torch.Tensor,torch.Tensor]):
        if index < 0:
            raise ValueError("Negative indices are not supported.")
        # Image
        X = self.image[index]
        # Label
        y = self.label[index]

        print("Avant",X.shape)
        X_norm = percentile(X, self.normalization)

        print("Après :", X_norm.shape)



# DataAugmentation
class AugmentationConfig(BaseModel):
    """Data augmentation parameters."""

    model_config = ConfigDict(extra="forbid")

    # Spatial augmentation
    do_affine_transform: bool = False
    affine_transform_prob: float = 0.5 # proba 0.5
    rotate_range: float = math.pi / 6 # rotation de 30 degré
    scale_range: tuple[float, float] = (0.85, 1.15) # changement de taille 85% et 115%

    # Gaussian noise
    do_gaussian_noise: bool = False
    gaussian_noise_prob: float = 0.2 # proba 0.2
    gaussian_noise_mean: float = 0.0 # mean center around 0
    gaussian_noise_std: float = 0.05 # std around 0.05

    # Contrast adjustment
    do_adjust_contrast: bool = False
    adjust_contrast_prob: float = 0.2 # proba 0.2
    contrast_gamma_range: tuple[float, float] = (0.8, 1.2) 
    # gamma < 1 Image plus claire
    # gamma > 1 Image plus sombre


class DataAugmentor2D(nn.Module):
    """2D image augmentation for classification."""

    def __init__(
        self,
        config: AugmentationConfig | None = None,
    ):
        super().__init__()

        cfg = config if config is not None else AugmentationConfig()

        transforms = []

        if cfg.do_affine_transform:
            transforms.append(
                RandAffined(
                    keys=["image"],
                    prob=cfg.affine_transform_prob,
                    rotate_range=cfg.rotate_range,
                    scale_range=cfg.scale_range,
                    mode="bilinear",
                    padding_mode="reflection",
                )
            )

        if cfg.do_gaussian_noise:
            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    prob=cfg.gaussian_noise_prob,
                    mean=cfg.gaussian_noise_mean,
                    std=cfg.gaussian_noise_std,
                )
            )

        if cfg.do_adjust_contrast:
            transforms.append(
                RandAdjustContrastd(
                    keys=["image"],
                    prob=cfg.adjust_contrast_prob,
                    gamma=cfg.contrast_gamma_range,
                )
            )

        self.transforms = Compose(transforms)
        # Compose : Pipeline to add different successive transform

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:

        data = {"image": input_tensor}

        data = self.transforms(data)

        return data["image"]