import os
from typing import Any, Callable, List, Optional, Tuple
import torchvision
from PIL import Image
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

from utils import get_file_names

class LabeledDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, dataset_info: pd.DataFrame,
     transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
       target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.transforms = transforms
        self.dataset_info = dataset_info
        self.rootdir = root
        print("Length of dataset is ", len(self.dataset_info))
        #self.images_labels = self.get_pil_imgs(self.dataset_info)
        

    def get_pil_imgs(self, df: pd.DataFrame):
        df  = df.reset_index()
        images_labels_pairs = []
        for index, row in tqdm(df.iterrows()):
            img_path = row['sample']
            label = row['label']
            img = self.pil_loader(os.path.join(self.rootdir, img_path))
            img_transformed = self.transforms(img)
            img.close()
            images_labels_pairs.append((img_transformed,label))
        return images_labels_pairs

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index: int) -> Any:
        img_path = self.dataset_info.iloc[index]['sample']
        label = self.dataset_info.iloc[index]['label']
        img = self.pil_loader(os.path.join(self.rootdir, img_path))
        return self.transforms(img), label, index
        # img, label = self.images_labels[index]
        # return img, label

    def __len__(self) -> int:
        return len(self.dataset_info)

def get_img_key(img_name: str):
    tokens = img_name.split(".jpeg")
    return int(tokens[0])
class UnlabeledDataset(torchvision.datasets.VisionDataset):
    def __init__(self, rootdir: str,
     transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
       target_transform: Optional[Callable] = None) -> None:
        super().__init__(rootdir, transforms, transform, target_transform)
        self.transforms = transforms
        self.rootdir = rootdir
        self.images_names = get_file_names(rootdir)
        self.images_names.sort(key = get_img_key)
        print("Length of dataset is ", len(self.images_names))
        #self.images_labels = self.get_pil_imgs(self.dataset_info)
        
    
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index: int) -> Any:
        img = self.pil_loader(os.path.join(self.rootdir, self.images_names[index]))
        return self.transforms(img), self.images_names[index]
    def __len__(self) -> int:
        return len(self.images_names)
class ContrastiveDataset(torchvision.datasets.VisionDataset):
    def __init__(self, images: List[str],
     transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
       target_transform: Optional[Callable] = None) -> None:
        self.transforms = transforms
        self.files = images
        print("Length of dataset is ", len(self.files))
        

    def get_file_names(self, rootdir):
        return [f for f in listdir(rootdir) if isfile(join(rootdir, f))]

    def get_pil_imgs(self, df: pd.DataFrame):
        df  = df.reset_index()
        images_labels_pairs = []
        for index, row in tqdm(df.iterrows()):
            img_path = row['sample']
            label = row['label']
            img = self.pil_loader(os.path.join(self.rootdir, img_path))
            img_transformed = self.transforms(img)
            img.close()
            images_labels_pairs.append((img_transformed,label))
        return images_labels_pairs

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index: int) -> Any:
        img = self.pil_loader(self.files[index])
        return self.transforms(img)
        # img, label = self.images_labels[index]
        # return img, label

    def __len__(self) -> int:
        return len(self.files)