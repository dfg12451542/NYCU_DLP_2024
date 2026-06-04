import argparse
import os
import numpy as np
import math

import torchvision.transforms as transformsfrom 
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
from tqdm import tqdm
from torchvision.utils import save_image

from dataset import iclevrDataset
from GAN import GAN_generate,GAN_discriminate

from evaluator import evaluation_model

#import torchsummary

def train():
    # torch.cuda.empty_cache()
    # torch.cuda.memory_summary(device=None, abbreviated=False)

    args = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 400
    batch_size = 128

    generator = GAN_generate(args).to(device)
    discriminator = GAN_discriminate(args).to(device)
    evaluator = evaluation_model()

    #generator.load_state_dict(torch.load('check_1/GAN_generator_epoch_350.pt'))
    #discriminator.load_state_dict(torch.load('check_1/GAN_discriminator_epoch_350.pt'))

    # print(sum(p.numel() for p in generator.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

    lr = 1e-4
    betas = (0.5,0.999)

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    train_dataloader = DataLoader(iclevrDataset(mode='train', root='iclevr'),batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(iclevrDataset(mode='test', root='iclevr'),batch_size=1,shuffle=True)

    criterion = nn.BCELoss()

    acc_vec = []
    generator_loss_vec = []
    discriminator_loss_vec = []

    best_acc = 0

    for i in range(epochs):
        discriminator.train()
        generator.train()

        generator_loss_total = 0
        discriminator_loss_total = 0

        counter = 0
        for (image, cond) in tqdm(train_dataloader):
            counter += 1
            #print(counter)
            if counter >= len(train_dataloader) :
                break
            # print(cond.shape)
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            image = image.to(device)
            cond = cond.to(device)

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            generate_img = generator(noise, cond)

            #while(1):a=1

            is_flip = False
            if np.random.random() < 0.1:
                image, generate_img = generate_img, image
                is_flip = True

            output_real = discriminator(image, cond)
            output_fake = discriminator(generate_img.detach(), cond)

            real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)
            fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)

            real_loss = criterion(output_real, real_label)
            fake_loss = criterion(output_fake, fake_label)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            generator_label = torch.ones(batch_size).to(device)
            output_fake = discriminator(generate_img, cond)

            generator_loss = criterion(output_fake, generator_label)
            generator_loss.backward()
            optimizer_generator.step()

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()

        generator_loss_vec.append(generator_loss_total/(len(train_dataloader) -1))
        discriminator_loss_vec.append(discriminator_loss_total/(len(train_dataloader) -1))

        print('epoch:',i,' generator_loss:',generator_loss_total/(len(train_dataloader) -1),' discriminator_loss:',discriminator_loss_total/(len(train_dataloader) -1))

        discriminator.eval()
        generator.eval()
        acc_total = 0
        with torch.no_grad():
            for (cond) in tqdm(test_dataloader):
                cond = cond.to(device)
                noise = torch.randn(1, 100, 1, 1, device=device)
                generate_img = generator(noise, cond)

                acc = evaluator.eval(generate_img,cond)
                acc_total += acc

            acc_vec.append(acc_total/(len(test_dataloader) ))
            print('acc:',acc_total/(len(test_dataloader) ))

            dfs = {
                'generator_loss_vec':generator_loss_vec,
                'discriminator_loss_vec':discriminator_loss_vec,
                'acc_vec':acc_vec
            }
            df = pd.DataFrame(dfs)
            df.to_csv('out.csv', index=False)  

            if (i+1)%50 == 0:
                torch.save(generator.state_dict(), os.path.join("check",f"GAN_generator_epoch_{i+1}.pt"))
                torch.save(discriminator.state_dict(), os.path.join("check",f"GAN_discriminator_epoch_{i+1}.pt"))

            if acc_total/(len(test_dataloader) ) > best_acc:
                best_acc = acc
                torch.save(generator.state_dict(), os.path.join("check","GAN_best_generator.pt"))
                torch.save(discriminator.state_dict(), os.path.join("check","GAN_best_discriminator.pt"))


if __name__ == '__main__':
    train()







