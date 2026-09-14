from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


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

        X_norm = percentile(X, self.normalization)

        return torch.tensor(X_norm).unsqueeze(0), torch.tensor(y)


        
        
