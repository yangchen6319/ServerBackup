from email.mime import image
from logging import root
import os
from queue import PriorityQueue
import torch
import numpy as np
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root,'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        # 得到image和mask的路径
        img_path = os.path.join(self.root,'PNGImages',self.imgs[idx])
        mask_path = os.path.join(self.root,'PedMasks',self.masks[idx])
        # 读取img和mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        print(mask)

pennFudanDataset = PennFudanDataset('data/PennFudanPed', 'transforms')
pennFudanDataset[1]