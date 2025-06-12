from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pandas import read_csv
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


class DatasetSplit(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class WhoWeAreDataset(Dataset):
    def __init__(self, annotations_dir: Path, img_dir: Path, 
                 split: DatasetSplit, transform: Optional[Callable] = None):
        
        annotations_path = Path(annotations_dir, split.value, 'annotations.csv')
        self.data_dir = Path(img_dir, split.value)
        self.annotations = read_csv(annotations_path, delimiter=',',
                                    names=['Image_Path', 'Disease'],
                                    dtype={0: object, 1: np.int32})
        self.transform = transform
 
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        image = decode_image(img_path).float()
        
        if self.transform:
            image = self.transform(image)

        image /= 255.

        return image, label