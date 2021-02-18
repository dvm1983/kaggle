import cv2
import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

#img_size = 300


def get_train_transforms(img_size):
    return Compose([
            RandomResizedCrop(img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
#            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
#            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms(img_size):
    return Compose([
            CenterCrop(img_size, img_size, p=1.),
            Resize(img_size, img_size),
#            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#            ToTensorV2(p=1.0),
        ], p=1.)


def sampling(df, sample_coeffs, random_state):
    df = pd.concat([df[df['label']==label].sample(frac=coeff, random_state=random_state, replace=True) for label, coeff in sample_coeffs.items()])
    return df.sample(frac=1, random_state=random_state)


means = np.array([[[0.485, 0.456, 0.406]]])
stds = np.array([[[0.229, 0.224, 0.225]]])

class MyDataset(Dataset):
    def __init__(self, train_df=None, train_idx=None, train_path=None,
                             train_df2=None, train_path2=None,
                             path_dct=None, path_extra=None, extra_df=None, path_extra2=None, img_size=300, shuffle=0,
                             augmentations=None, augmentations2=None, test_time=False, two_samples=False,
                             advprop=False, tta=None, sample_coeffs=None, fullsampling=True):
        self.augmentations = augmentations
        self.augmentations2 = augmentations2
#        self.test_time = test_time
        self.tta = tta
        self.img_size = img_size

        self.two_samples = False
#        if path_extra is not None:
        self.two_samples = two_samples # True

        if advprop:
            self.normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
        else:
            self.normalize = lambda img: (img/255. - means)/stds
#            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                   std=[0.229, 0.224, 0.225])        
        self.data = None
        self.labels = None
        
        if train_idx is not None:
            train_df = train_df.copy()
            train_df = train_df.iloc[train_idx]
        
        if train_df is not None:
            train_df['image_id'] = train_df['image_id'].apply(lambda x: f'{train_path}/{x}')
            
        if train_df2 is not None:
            train_df2 = train_df2.copy()
            train_df2['image_id'] = train_df2['image_id'].apply(lambda x: f'{train_path2}/{x}')
            train_df2 = train_df2.sample(frac=1, random_state=shuffle).reset_index(drop=True)
            train_df = pd.concat([train_df, train_df2])
        
        if (not fullsampling) and sample_coeffs is not None:
            train_df = sampling(train_df, sample_coeffs, random_state=shuffle)
        
        if path_dct is not None:
            df_lst = []
            for label in path_dct:
                path = path_dct[label]
                files = [f'{path}/{x}' for x in os.listdir(path) if x.split('.')[-1] == 'jpg']
                df = pd.DataFrame(files, columns=['image_id'])
                df['label'] = [label]*len(files)
                df_lst.append(df)
            if train_df is not None:
                df_lst.append(train_df)
            train_df = pd.concat(df_lst)
        
        if fullsampling and sample_coeffs is not None:
            train_df = sampling(train_df, sample_coeffs, random_state=shuffle)            
        
        if train_df is not None and shuffle is not None:
            train_df = train_df.sample(frac=1, random_state=shuffle).reset_index(drop=True)    

        if path_extra is None:
            self.labels = list(train_df['label'])
            self.data = list(train_df['image_id'])
        else:
            if extra_df is None:
                self.data = [f'{path_extra}/{x}' for x in os.listdir(path_extra) if x.split('.')[-1] == 'jpg']
            else:
                self.data = [f'{path_extra}/{x}' for x in extra_df['image_id'].values]
            if path_extra2 is not None:
                self.data += [f'{path_extra2}/{x}' for x in os.listdir(path_extra2) if x.split('.')[-1] == 'jpg']
        
    def __getitem__(self, idx):
        img_bgr = cv2.imread(f"{self.data[idx]}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        if self.augmentations:
            if self.two_samples:
                img = self.augmentations(image=img)['image']
                img2 = self.augmentations2(image=img)['image']
            else:
                if self.tta is None:
                    img = self.augmentations(image=img)['image'] 
                else:
                    img = [img.copy() for i in range(self.tta)]
                    img = [self.augmentations(image=x)['image'] for x in img]
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.tta is None:
            img = self.normalize(img)    
    #       img = (img/255. - means)/stds
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()#.unsqueeze(0).float()
        else:
            img = [torch.from_numpy(np.transpose(self.normalize(x), (2, 0, 1))).float() for x in img]
    
    
        if self.two_samples: 
            img2 = self.normalize(img2)
            img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).float()
        
        if self.labels is None:
            if self.two_samples:
                return img, img2 #with aug, witout aug
            else:
                return img
        else:
            return img, self.labels[idx]
        
    def __len__(self):
        return len(self.data)