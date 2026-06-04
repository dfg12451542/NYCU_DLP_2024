import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import pandas as pd
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self,train_loader,step):
        step = step
        loss_total = 0
        for imgs in tqdm(train_loader):
            imgs = imgs.to(device=args.device)
            logits, target = self.model(imgs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_total += loss.item()
            loss.backward()
            
            if step % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            step+=1
            self.scheduler.step()
        #pass
        return loss_total/len(train_loader)

    def eval_one_epoch(self,val_loader):
        loss_total = 0
        for imgs in tqdm(val_loader):
            imgs = imgs.to(device=args.device)
            logits, target = self.model(imgs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_total += loss.item()
        #pass
        return loss_total/len(val_loader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.1)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    eval_vec = []
    train_vec = []
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch {epoch}:")
        step = args.start_from_epoch * len(train_loader)
        loss_train = train_transformer.train_one_epoch(train_loader,step)
        #loss_eval = train_transformer.eval_one_epoch(val_loader)
        loss_eval = 0

        print('loss_train: ',loss_train,' loss_eval: ',loss_eval)

        eval_vec.append(loss_eval)
        train_vec.append(loss_train)

        dfs = {
            'loss_vec':eval_vec,
            'train_vec':train_vec,
        }
        df = pd.DataFrame(dfs)
        df.to_csv('out.csv', index=False)  

        if epoch % args.ckpt_interval == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", f"transformer_epoch_{epoch}.pt"))
        torch.save(train_transformer.model.transformer.state_dict(), os.path.join("transformer_checkpoints", "transformer_current.pt"))

        #pass