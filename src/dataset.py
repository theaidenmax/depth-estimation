import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path

class NYUDepthDataset(Dataset):
    def __init__(self, csv_file, transform_img=None, transform_depth=None):
        self.csv_path = Path(csv_file)
        
        self.root_dir = self.csv_path.parent 
    
        self.df = pd.read_csv(self.csv_path, header=None)
        
        self.transform_img = transform_img
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.df)

    def _fix_path(self, raw_path):
        clean_path_str = str(raw_path).replace('\\', '/')
        
        path_obj = Path(clean_path_str)
        if path_obj.parts[0] == 'data':
            path_obj = Path(*path_obj.parts[1:])
            
        full_path = self.root_dir / path_obj
        return full_path

    def __getitem__(self, idx):
        img_raw_path = self.df.iloc[idx, 0]
        depth_raw_path = self.df.iloc[idx, 1]

        img_path = self._fix_path(img_raw_path)
        depth_path = self._fix_path(depth_raw_path)

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        return image, depth