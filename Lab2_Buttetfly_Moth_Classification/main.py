import warnings
warnings.filterwarnings('ignore')

from dataloader import BufferflyMothLoader
from torch.utils.data import TensorDataset, DataLoader

from VGG19 import my_network
#from ResNet50 import my_network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate(net,valid_loader):
    #rint("evaluate() not defined")
    net.eval()
    criterion = nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        valid_loss = 0
        correct = 0
        total = 0
        for valid_data in valid_loader:
            imgs, labels = valid_data
            imgs, labels = imgs.to(device), labels.to(device)
            pred_labels = net(imgs)
            loss = criterion(pred_labels, labels)
            _, maxID = torch.max(pred_labels.data, 1)
            total += labels.size(0)
            for ans, pred in zip(labels, maxID):
                if pred == ans:
                    correct += 1
            valid_loss += loss.item()
        valid_acc = correct / total * 100
        valid_loss = valid_loss / len(valid_loader)
        #print('valid loss: %.3f valid acc: %.3f' % \
        #    (valid_loss, valid_acc))
        return valid_loss, valid_acc

def test():
    #print("test() not defined")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = my_network().to(device)
    model.load_state_dict(torch.load('w_312551097_VGG_1.pth'))

    batch_size = 16
    test_dataset = BufferflyMothLoader(root = 'dataset/',mode = 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for test_data in test_loader:
            imgs, labels = test_data
            imgs, labels = imgs.to(device), labels.to(device)
            pred_labels = model(imgs)
            loss = criterion(pred_labels, labels)
            _, maxID = torch.max(pred_labels.data, 1)
            total += labels.size(0)
            for ans, pred in zip(labels, maxID):
                if pred == ans:
                    correct += 1
            test_loss += loss.item()
        test_acc = (correct / total)*100
        test_loss = test_loss / len(test_loader)
        print('test loss: %.3f test acc: %.3f' % \
            (test_loss, test_acc))
        #return test_loss, test_acc

def train():
    #print("train() not defined")
    batch_size = 16
    train_dataset = BufferflyMothLoader(root = 'dataset/',mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    print(len(train_loader))

    valid_dataset = BufferflyMothLoader(root = 'dataset/',mode = 'valid')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = my_network().to(device)
    print('GPU state:', device)

    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
    epochs = 30

    model_save_path = 'w_312551097.pth_test_V'

    train_acc_value = []
    train_loss_value = []
    valid_acc_value = []
    valid_loss_value = []

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        print()
        for train_data in train_loader:
            net.train()
            imgs, labels = train_data
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred_labels = net(imgs)
            loss = criterion(pred_labels, labels)
            _, maxID = torch.max(pred_labels.data, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(labels)
            #print(maxID)
            for ans, pred in zip(labels, maxID):
                if pred == ans:
                    correct += 1
            total += labels.size(0)
            #print(labels.size(0))

        train_loss = train_loss / len(train_loader)
        train_acc = (correct / total)*100
        #print(correct)
        #print(total)

        #exp_lr_scheduler.step()

        valid_loss, valid_acc = evaluate(net,valid_loader)

        print('epoch %2d: train loss: %.3f train acc: %.3f valid loss: %.3f valid acc: %.3f' % \
        (epoch+1, train_loss, train_acc, valid_loss, valid_acc))

        train_acc_value.append(train_acc)
        train_loss_value.append(train_loss)
        valid_acc_value.append(valid_acc)
        valid_loss_value.append(valid_loss)
            
        if (epoch+1) % 10 == 0:    
            torch.save(net.state_dict(),model_save_path)

    plt.plot(train_loss_value,color = 'blue', linewidth=2, marker='o')
    plt.plot(valid_loss_value,color = 'red', linewidth=2, marker='o')
    plt.savefig('plot_loss.png') 
    plt.show()

    plt.plot(train_acc_value,color = 'blue', linewidth=2, marker='o')
    plt.plot(valid_acc_value,color = 'red', linewidth=2, marker='o')
    plt.savefig('plot_acc.png') 
    plt.show()



if __name__ == "__main__":
    #print("Good Luck :)")
    #train()
    test()