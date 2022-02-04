# file that scans for data from ordered folders and generates DataLoader class.
# -----------------------------------------------------------------------------------------------
# Classification - data/train/images, data/train/data.csv, data/val/images and data/val/data.csv.
# Segmentation - data/train/images, data/train/masks, data/val/images and data/val/masks.
# Object Detection - data/train/images, data/train/data.csv, data/val/images and data/val/data.csv.
# bboxes - [x, y, h, w, classes]

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from PIL import Image
from utils import *
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


class ImageDataset(Dataset):
    def __init__(self, path, mode, device, transforms=None, train=True):
        super().__init__()

        self.transforms, self.mode, self.device = transforms, mode, device
        self.path = path
        
        self.df = pd.read_csv(f"{self.path}\\data.csv")

        if mode == "classification":
            self.classes = self.df["class"].unique()
            self.df[self.classes] = pd.get_dummies(self.df["class"])
            del self.df["class"]

        if mode == "detection":
            for i in range(len(self.df)):
                self.df["labels"].iloc[i] = get_boxxes(self.df["labels"].iloc[i])

    def __getitem__(self, idx):

        img_path = f"{self.path}\\images\\" + self.df["img_path"].iloc[idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.mode == "classification":

            label = torch.tensor(np.array(self.df[self.classes].iloc[idx]))

            if self.transforms:

                img = self.transforms["image"]

        elif self.mode == "detection":

            label = np.array(self.df["labels"].iloc[idx])

            if self.transforms:

                img = self.transforms["image"]
                label = self.transforms["bboxes"]

            label = torch.tensor(label)

        elif self.mode == "segmentation":

            img_path = f"{self.path}/masks/" + self.df["mask_path"].iloc[idx]
            label = np.array(Image.open(img_path).convert("RGB"))

            if self.transforms:

                img = self.transforms["image"]
                label = self.transforms["mask"]

            label = torch.tensor(label).permute(2, 0, 1)

        return (
            torch.tensor(img).permute(2, 0, 1).float().to(self.device),
            label.float().to(self.device),
        )

    def __len__(self):
        return len(self.df)


def get_dataset(path, mode, transforms= None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainset = ImageDataset(f"{path}\\train\\", mode, device, transforms= transforms, train=True)
    valset = ImageDataset(f"{path}\\val\\", mode, device, train=False)

    return trainset, valset
