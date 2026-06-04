import argparse
import os
import numpy as np
import math

import torchvision.transforms as transformsfrom 
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image

from dataset import iclevrDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()

    train_dataloader = DataLoader(
        iclevrDataset(mode='train', root='iclevr'),
        batch_size=1,
        shuffle=True
    )

    for i, (image, cond) in enumerate(train_dataloader):
        if i == 1:
            save_image(image,'real_samples.png',normalize=True)
            break
