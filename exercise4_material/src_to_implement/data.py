from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from torchvision.transforms import v2

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self,data,model):
        self.data = data
        self.model=model
        self._transform=v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomRotation(90),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(train_mean, train_std),
        ])
        self._transform_val=v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(train_mean, train_std),
        ])
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        row = self.data.iloc[index]
        img_path = Path(row['filename'])
        is_crack ,is_inactive= row['crack'],row['inactive']
        image = gray2rgb(imread(img_path))
        if self.model == 'val':
            img = self._transform_val(image)
        else:
            img = self._transform(image)
        return img,torch.tensor([is_crack, is_inactive], dtype=torch.float) 


        
