import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)

        self.head_layer = self.sample(3,64)
        self.down_layer_1 = self.sample(64,64*2)
        self.down_layer_2 = self.sample(64*2,64*4)
        self.down_layer_3 = self.sample(64*4,64*8)
        self.down_layer_4 = self.sample(64*8,64*16)

        self.up_layer_1 = self.sample(64*16,64*8)
        self.up_layer_2 = self.sample(64*8,64*4)
        self.up_layer_3 = self.sample(64*4,64*2)
        self.up_layer_4 = self.sample(64*2,64)

        self.up1 = nn.ConvTranspose2d(64*16, 64*8 , kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(64*8, 64*4 , kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64*4, 64*2 , kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64*2, 64 , kernel_size=2, stride=2)

        self.conv = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_1 = self.head_layer(input)
        x_2 = self.down_layer_1(self.max_pool(x_1))
        x_3 = self.down_layer_2(self.max_pool(x_2))
        x_4 = self.down_layer_3(self.max_pool(x_3))
        x_5 = self.down_layer_4(self.max_pool(x_4))
        
        x = self.up1(x_5)
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

        x = self.conv(x)

        return torch.sigmoid(x)

    def sample(self, input_size, output_size):
        layers = []

        layers.append(nn.Conv2d(input_size, output_size, 3, 1,1))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(output_size, output_size, 3, 1,1))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

