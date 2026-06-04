import argparse
import os
import numpy as np
import math

import torchvision.transforms as transformsfrom 
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch

class GAN_generate(nn.Module):
    def __init__(self, args):
        super(GAN_generate, self).__init__()
        self.class_size = 24
        self.condition_size = 100
        self.latent_size = 100
        self.generate_size = 300

        # self.linear = nn.Linear(self.class_size, self.condition_size)
        # self.leak = nn.LeakyReLU(0.2, True)

        self.label_emb = nn.Sequential(
            nn.Linear(self.class_size, self.condition_size),
            nn.LeakyReLU(0.2, True)
        )

        # self.conv1 = nn.ConvTranspose2d(self.latent_size + self.condition_size, self.generate_size * 8, 4, 1, 0, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.generate_size * 8)

        # self.conv2 = nn.ConvTranspose2d(self.generate_size * 8, self.generate_size * 4, 4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(self.generate_size * 4)

        # self.conv3 = nn.ConvTranspose2d(self.generate_size * 4, self.generate_size * 2, 4, 2, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.generate_size * 2)

        # self.conv4 = nn.ConvTranspose2d(self.generate_size * 2, self.generate_size * 1, 4, 2, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(self.generate_size * 1)

        # self.conv5 = nn.ConvTranspose2d(self.generate_size * 1, 3, 4, 2, 1, bias=False)

        # self.relu = nn.ReLU(True)
        # self.tanh = nn.Tanh()

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size + self.condition_size, self.generate_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.generate_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generate_size * 8, self.generate_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generate_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generate_size * 4, self.generate_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generate_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generate_size * 2, self.generate_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generate_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generate_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise_latent, labels):
        # print(labels.shape)
        # label_emb = self.leak(self.linear(labels)).view(-1, self.condition_size, 1, 1)
        # print(label_emb.shape)
        # print(noise_latent.shape)

        label_emb = self.label_emb(labels).view(-1, self.condition_size, 1, 1)

        input = torch.cat((label_emb, noise_latent), 1)

        out = self.seq(input)

        # out = self.relu(self.bn1(self.conv1(input)))
        # out = self.relu(self.bn2(self.conv2(out)))
        # out = self.relu(self.bn3(self.conv3(out)))
        # out = self.relu(self.bn4(self.conv4(out)))
        # out = self.tanh(self.conv5(out))

        return out
    
class GAN_discriminate(nn.Module):
    def __init__(self, args):
        super(GAN_discriminate, self).__init__()
        self.class_size = 24
        self.condition_size = 100
        self.latent_size = 100
        self.discriminate_size = 100
        self.img_size = 64

        # self.linear = nn.Linear(self.class_size, self.img_size*self.img_size)
        # self.leak = nn.LeakyReLU(0.2, True)

        self.label_emb = nn.Sequential(
            nn.Linear(self.class_size, self.img_size * self.img_size),
            nn.LeakyReLU(0.2, True)
        )

        # self.conv1 = nn.Conv2d(3+1, self.discriminate_size * 1, 4, 2, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.discriminate_size * 1)

        # self.conv2 = nn.Conv2d(self.discriminate_size * 1, self.discriminate_size * 2, 4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(self.discriminate_size * 2)

        # self.conv3 = nn.Conv2d(self.discriminate_size * 2, self.discriminate_size * 4, 4, 2, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.discriminate_size * 4)

        # self.conv4 = nn.Conv2d(self.discriminate_size * 4, self.discriminate_size * 8, 4, 2, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(self.discriminate_size * 8)

        # self.conv5 = nn.Conv2d(self.discriminate_size * 8, 1, 4, 1, 0, bias=False)
        # self.sigmoid = nn.Sigmoid()

        self.seq = nn.Sequential(
            nn.Conv2d(3 + 1, self.discriminate_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminate_size, self.discriminate_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminate_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminate_size * 2, self.discriminate_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminate_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminate_size * 4, self.discriminate_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminate_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminate_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # label_emb = self.leak(self.linear(labels)).view(-1, 1, self.img_size, self.img_size)

        label_emb = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        input = torch.cat((img, label_emb), dim=1)

        #print(input.shape)

        out = self.seq(input)

        #out = self.leak(self.bn1(self.conv1(input)))
        # out = self.leak((self.conv1(input)))
        #print(out.shape)
        # out = self.leak(self.bn2(self.conv2(out)))
        # out = self.leak(self.bn3(self.conv3(out)))
        # out = self.leak(self.bn4(self.conv4(out)))
        # out = self.sigmoid(self.conv5(out))

        #print(out.shape)

        return out.view(-1, 1).squeeze(1)