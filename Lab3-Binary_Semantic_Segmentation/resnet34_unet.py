# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models

class Bottleneck_1(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck_1, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
          identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return(x)
    
class ResnetUnet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResnetUnet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.up_layer_1 = self.sample(256+512,64)
        self.up_layer_2 = self.sample(32+256,64)
        self.up_layer_3 = self.sample(32+128,64)
        self.up_layer_4 = self.sample(32+64,64)
        self.up_layer_44 = self.sample(128,64)

        self.body = nn.Sequential(nn.Conv2d(512,1024,kernel_size = 3, stride=1, padding=1),nn.BatchNorm2d(1024),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size = 2, stride=2, padding=0))
        
        self.up1 = nn.ConvTranspose2d(1024, 256 , kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 32 , kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32 , kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32 , kernel_size=2, stride=2)
        self.up44 = nn.ConvTranspose2d(64, 64 , kernel_size=2, stride=2)
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64, 64 , kernel_size=2, stride=2))

        self.conv = nn.Conv2d(64, 1, 1)

        self.num_classes = num_classes
        
    def forward(self, x):
        #print(x.shape)
        x = self.batch_norm1(self.conv1(x))
        x_44 = self.relu(x)
        x = self.max_pool(x_44)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        x = self.body(x_4)

        x = self.up1(x)
        x = torch.concat((x,x_4),dim = 1)
        x = self.up_layer_1(x)

        x = self.up2(x)
        x = torch.concat((x,x_3),dim = 1)
        x = self.up_layer_2(x)

        x = self.up3(x)
        x = torch.concat((x,x_2),dim = 1)
        x = self.up_layer_3(x)

        x = self.up4(x)
        x = torch.concat((x,x_1),dim = 1)
        x = self.up_layer_4(x)

        x = self.up44(x)
        x = torch.concat((x,x_44),dim = 1)
        x = self.up_layer_44(x)

        x = self.up5(x)

        x = self.conv(x)
        
        return torch.sigmoid(x)
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if self.in_channels != planes*1:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*1, kernel_size=1, stride=stride, bias=False),
                #nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
                nn.BatchNorm2d(planes*1)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*1
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
    def sample(self, input_size, output_size):
        layers = []

        layers.append(nn.Conv2d(input_size, output_size, 3, 1,1))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(output_size, output_size, 3, 1,1))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
def my_network(num_classes=100, channels=3):
    return ResnetUnet(Bottleneck_1, [3,4,6,3], num_classes, channels)