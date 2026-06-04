import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        df = pd.read_csv('dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.        
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + self.img_name[index]
        img = Image.open(path)
        transforms = [
            torchvision.transforms.GaussianBlur(7,3),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.Resize((200, 200)),
            #torchvision.transforms.RandomResizedCrop((200, 200)),
            #torchvision.transforms.RandomRotation(30, expand=False),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
        ]
        transforms_ = [
            torchvision.transforms.GaussianBlur(7,3),
            torchvision.transforms.Resize((200, 200)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
        ]

        trans = torchvision.transforms.Compose(transforms)
        trans_ = torchvision.transforms.Compose(transforms_)

        if self.mode == 'train':
            img = trans(img)
        else:
            img = trans_(img)

        label = self.label[index]

        #print(img)

        return img, label
