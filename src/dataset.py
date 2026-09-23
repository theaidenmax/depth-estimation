import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, transform_img=None, transform_depth=None):
        self.df = pd.read_csv(csv_file, header=None)
        self.transform_img = transform_img
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        depth_path = self.df.iloc[idx, 1]

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        return image, depth