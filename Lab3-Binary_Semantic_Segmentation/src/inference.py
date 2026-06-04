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
import cv2

from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import my_network
from evaluate import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--net',  type=str, default='unet', help='net type')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.net == 'unet':
        net = UNet().to(device)
    else:
        net = my_network().to(device)

    net.load_state_dict(torch.load(args.model))

    testloss,test_score = evaluate(net,test_loader)
    print('test loss : ',testloss/len(test_loader),' test score : ',test_score/len(test_loader))

    for valid_data in test_loader:
        imgs, masks, trimaps = valid_data['image'], valid_data['mask'], valid_data['trimap']
        imgs, masks = imgs.to(device), masks.to(device)
        imgs = imgs.to(torch.float)
        pred_masks = net(imgs)

        pred_masks = pred_masks.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy()
        imgs = imgs.cpu().detach().numpy()


        img = np.transpose(imgs[0],(1,2,0))
        #img = np.uint8(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mask = np.transpose(masks[0]*256,(1,2,0))
        #mask = np.uint8(mask)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

        pred_mask = np.transpose(pred_masks[0]*256,(1,2,0))
        pred_mask = np.uint8(pred_mask)
        pred_mask = cv2.cvtColor(pred_mask,cv2.COLOR_BGR2RGB)

        cv2.imwrite('test.png', mask)
        #print(mask)

        for i in range(256):
            for j in range(256):
                if mask[i][j][0] == 256:
                    mask[i][j][0] = img[i][j][0]
                    mask[i][j][1] = img[i][j][1]
                    mask[i][j][2] = img[i][j][2]

                if pred_mask[i][j][0] > 100:
                    pred_mask[i][j][0] = img[i][j][0]
                    pred_mask[i][j][1] = img[i][j][1]
                    pred_mask[i][j][2] = img[i][j][2]

        cv2.imwrite('img_original.png', img)
        cv2.imwrite('img_mask.png', mask)
        cv2.imwrite('img_pred_mask.png', pred_mask)

        break

    #assert False, "Not implemented yet!"