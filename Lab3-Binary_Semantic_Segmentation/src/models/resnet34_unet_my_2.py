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

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
          #print('a')
          #print(identity.shape)
          identity = self.i_downsample(identity)
        # print('skip')
        # print(x.shape)
        # print(identity.shape)
        # print('end')
        x += identity
        x = self.relu(x)

        return(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # identity or not (projected mapping)
        if in_channels == out_channels:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False))
            projection_layer.append(nn.BatchNorm2d(out_channels))
            self.projection = nn.Sequential(*projection_layer)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        in_x = x
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)
        
        x = self.relu(x)

        return x

class ResnetUnet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResnetUnet, self).__init__()
        self.in_channels = 64
        channel_list = [64, 128, 256, 512]
        repeatition_list = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.body = nn.Sequential(
            nn.Conv2d(channel_list[3], channel_list[3]*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel_list[3]*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        #self.body = nn.Sequential(nn.Conv2d(512,1024,kernel_size = 3, stride=1, padding=1),nn.BatchNorm2d(1024),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size = 2, stride=2, padding=0))

        self.decoder1 = DecoderBlock(conv_in_channels=channel_list[3]*2, conv_out_channels=channel_list[3])
        self.decoder2 = DecoderBlock(conv_in_channels=channel_list[3], conv_out_channels=channel_list[2])
        self.decoder3 = DecoderBlock(conv_in_channels=channel_list[2], conv_out_channels=channel_list[1])
        self.decoder4 = DecoderBlock(conv_in_channels=channel_list[1], conv_out_channels=channel_list[0])
        self.decoder5 = DecoderBlock(
            conv_in_channels=channel_list[1], conv_out_channels=channel_list[0], up_in_channels=channel_list[0], up_out_channels=channel_list[0]
        )

        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=2, stride=2),
            nn.Conv2d(channel_list[0], 1, kernel_size=3, padding=1, bias=False)
        )

        # self.up1 = nn.ConvTranspose2d(1024, 512 , kernel_size=2, stride=2)
        # self.up_layer_1 = self.sample(512+512,512)

        # self.up2 = nn.ConvTranspose2d(512, 256 , kernel_size=2, stride=2)
        # self.up_layer_2 = self.sample(256+256,256)

        # self.up3 = nn.ConvTranspose2d(256, 128 , kernel_size=2, stride=2)
        # self.up_layer_3 = self.sample(128+128,128)

        # self.up4 = nn.ConvTranspose2d(128, 64 , kernel_size=2, stride=2)
        # self.up_layer_4 = self.sample(64+64,64)

        # self.up44 = nn.ConvTranspose2d(64, 64 , kernel_size=2, stride=2)
        # self.up_layer_44 = self.sample(128,64)

        # self.up5 = nn.Sequential(nn.ConvTranspose2d(64, 64 , kernel_size=2, stride=2))

        # self.conv = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x = self.batch_norm1(self.conv1(x))
        x_44 = self.relu(x)
        x = self.max_pool(x_44)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        x = self.body(x_4)


        # x = self.up1(x)
        # x = torch.concat((x,x_4),dim = 1)
        # x = self.up_layer_1(x)

        # x = self.up2(x)
        # x = torch.concat((x,x_3),dim = 1)
        # x = self.up_layer_2(x)

        # x = self.up3(x)
        # x = torch.concat((x,x_2),dim = 1)
        # x = self.up_layer_3(x)

        # x = self.up4(x)
        # x = torch.concat((x,x_1),dim = 1)
        # x = self.up_layer_4(x)

        # x = self.up44(x)
        # x = torch.concat((x,x_44),dim = 1)
        # x = self.up_layer_44(x)

        # x = self.up5(x)

        # x = self.conv(x)

        d1 = self.decoder1(x, x_4)
        d2 = self.decoder2(d1, x_3)
        d3 = self.decoder3(d2, x_2)
        d4 = self.decoder4(d3, x_1)
        d5 = self.decoder5(d4, x_44)

        x = self.lastlayer(d5)
        
        return torch.sigmoid(x)
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*1:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*1, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*1)
            )
            
        #layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        layers.append(ResBlock(self.in_channels, planes, stride=stride))
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

    
class DecoderBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, up_in_channels=None, up_out_channels=None):
        super(DecoderBlock, self).__init__()

        if up_in_channels == None:
            up_in_channels = conv_in_channels
        if up_out_channels == None:
            up_out_channels = conv_out_channels

        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)

        return self.conv(x)
    
def my_network(num_classes=100, channels=3):
    return ResnetUnet(BasicBlock, [3,4,6,3],  channels)
    #return ResnetUnet(Bottleneck_1, [3,4,6,3],  channels)