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

from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import my_network
from evaluate import evaluate

def train(args):
    # implement the training function here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='train')
    valid_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='valid')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    if args.net == 'unet':
        net = UNet().to(device)
    else:
        net = my_network().to(device)

    print('GPU state:', device)

    criterion = nn.BCELoss()

    print(args)
    lr = args.learning_rate
    optimizer = optim.Adam(net.parameters(), lr=lr)
    epochs = args.epochs

    model_save_path = 'saved_models/w_312551097.pth_1'

    train_score_value = []
    train_loss_value = []
    valid_score_value = []
    valid_loss_value = []

    for epoch in range(epochs):
        train_loss = 0.0
        score_sum = 0.0
        total = 0.0
        for train_data in train_loader:
            net.train()
            imgs, masks, trimaps = train_data['image'], train_data['mask'], train_data['trimap']
            imgs, masks = imgs.to(device), masks.to(device)
            imgs = imgs.to(torch.float)
            # if_rotate = np.random.randint(2)
            # if_h = np.random.randint(2)
            # if_w = np.random.randint(2)
            # transforms = [
            #     torchvision.transforms.RandomHorizontalFlip(p=if_h),
            #     torchvision.transforms.RandomVerticalFlip(p=if_w),
            #     #torchvision.transforms.RandomRotation(30, expand=False),
            #     torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
            # ]
            # transforms_msak = [
            #     torchvision.transforms.RandomHorizontalFlip(p=if_h),
            #     torchvision.transforms.RandomVerticalFlip(p=if_w)
            # ]
            # trans = torchvision.transforms.Compose(transforms)
            # trsns_mask = torchvision.transforms.Compose(transforms_msak)
            # imgs = trans(imgs)
            # masks = trsns_mask(masks)
            optimizer.zero_grad()
            pred_masks = net(imgs)

            masks = torch.squeeze(masks,1)
            pred_masks = torch.squeeze(pred_masks,1)
 
            loss = criterion(pred_masks, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            score = dice_score(pred_masks.cpu().detach().numpy(), masks.cpu().detach().numpy())
            score_sum += score

            #print(len(train_loader))

        valid_loss,valid_score = evaluate(net,valid_loader)

        print('epoch : ',epoch,' train loss : ',train_loss/len(train_loader),' train score : ', score_sum/len(train_loader),' valid loss : ',valid_loss/len(valid_loader),' valid score : ', valid_score/len(valid_loader))
        
        train_score_value.append(score_sum/len(train_loader))
        train_loss_value.append(train_loss/len(train_loader))
        valid_score_value.append(valid_score/len(valid_loader))
        valid_loss_value.append(valid_loss/len(valid_loader))

        if (epoch+1) % 10 == 0:    
            torch.save(net.state_dict(),model_save_path)
    
    plt.plot(train_loss_value,color = 'blue', linewidth=2, marker='o')
    plt.plot(valid_loss_value,color = 'red', linewidth=2, marker='o')
    plt.savefig('plot_loss.png') 
    plt.show()

    plt.plot(train_score_value,color = 'blue', linewidth=2, marker='o')
    plt.plot(valid_score_value,color = 'red', linewidth=2, marker='o')
    plt.savefig('plot_acc.png') 
    plt.show()
    
    return NotImplemented

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--net',  type=str, default='unet', help='net type')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)