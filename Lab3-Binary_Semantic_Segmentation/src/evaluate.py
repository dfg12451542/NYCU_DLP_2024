import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision
from utils import dice_score
import numpy as np

def evaluate(net,valid_loader):
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()

    with torch.no_grad():
        valid_loss = 0.0
        score_sum = 0.0
        for valid_data in valid_loader:
            imgs, masks, trimaps = valid_data['image'], valid_data['mask'], valid_data['trimap']
            imgs, masks = imgs.to(device), masks.to(device)
            imgs = imgs.to(torch.float)

            # transforms = [
            #     torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
            # ]
            # trans = torchvision.transforms.Compose(transforms)
            # imgs = trans(imgs)

            pred_masks = net(imgs)

            masks = torch.squeeze(masks,1)
            pred_masks = torch.squeeze(pred_masks,1)

            loss = criterion(pred_masks, masks)
            valid_loss += loss.item()

            score = dice_score(pred_masks.cpu().detach().numpy(), masks.cpu().detach().numpy())
            score_sum += score

            #print(score)

    return valid_loss,score_sum