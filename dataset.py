from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from numpy import random
from torchvision import transforms
from glob import glob
import pandas as pd
import numpy as np
import albumentations
import albumentations.pytorch
import cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import os

def img_load(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))

    # sobelX = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cv2.CV_64F, 1, 0, ksize=5)
    # sobelX = cv2.convertScaleAbs(sobelX)
    # sobelY = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=5)
    # sobelY = cv2.convertScaleAbs(sobelY)
    # edge_img = cv2.addWeighted(sobelX, 1, sobelY, 1, 0)
    # result_img = cv2.addWeighted(edge_img, 1, img, 1, 0)

    # result_img = np.concatenate((img, edge_img), axis=-1)

    return img

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms = None, mode='train'):
        self.img_paths = img_paths
        # if labels != None: # train 할때는 주석
        self.labels = torch.LongTensor(labels)
        self.mode = mode
        self.transforms = transforms
        self.train_imgs = [img_load(m) for m in tqdm(self.img_paths)]
    # test_imgs = [img_load(n) for n in tqdm(test_png)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.train_imgs[idx]
        path = self.img_paths[idx]
        if transforms == None:
            img = img/255
        else:
            img = self.transforms(image=img)['image'] / 255.0


        if self.mode=='train':
            label = self.labels[idx]
            return img, label
        else :
            img_idx = int(path.replace('\\', '/').split('/')[-1].split('.')[0])
            img_idx = img_idx % 20000
            return str(img_idx), img


class DataModule(LightningDataModule):
    def __init__(self, batch_size = 12, test = False):
        super().__init__()
        self.save_hyperparameters()

        # self.train_png = np.array(sorted(glob('open/train/*.png')))
        self.test_png = sorted(glob('open/test/*.png'))

        # self.train_y = pd.read_csv("open/train_df.csv")
        self.train_y = pd.read_csv("open/train_df_aug.csv")
        self.x = self.train_y['file_name']
        train_labels = self.train_y["label"]
        self.label_unique = sorted(np.unique(train_labels))
        self.label_unique = {key: value for key, value in zip(self.label_unique, range(len(self.label_unique)))}
        self.train_labels = np.array([self.label_unique[k] for k in train_labels])

        fold_num = 0

        if not test:
            image_base_path = 'open/train'
            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
            for i, (train_index, valid_index) in enumerate(stratified_kfold.split(self.x, self.train_labels)):
                if i == fold_num:
                    # print(train_index,valid_index)
                    self.train_x, self.valid_x = self.x[train_index], self.x[valid_index]
                    self.train_y, self.valid_y = self.train_labels[train_index], self.train_labels[valid_index]
                    self.train_x = [os.path.join(image_base_path, file_name) for file_name in self.train_x]
                    self.valid_x = [os.path.join(image_base_path, file_name) for file_name in self.valid_x]
        else:
            self.X = self.test_png

        self.data_transforms = {
            'train': albumentations.Compose([
                albumentations.ColorJitter(p=0.7),
                albumentations.RandomBrightnessContrast(p=0.33),
                albumentations.GaussNoise(p=0.33),

                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=0.7),
                    albumentations.RandomRotate90(p=0.7),
                    albumentations.VerticalFlip(p=0.7),
                ], p=1),
# https: // tugot17.github.io / data - science - blog / albumentations / data - augmentation / tutorial / 2020 / 09 / 17 / Spatial - level - transforms - using - albumentations - package.html# ShiftScaleRotate
                albumentations.OneOf([
                    albumentations.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=0.7),
                    albumentations.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                    albumentations.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                ], p=1),

                # albumentations.Cutout(num_holes=16, max_h_size=15, max_w_size=15, fill_value=0),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),

                albumentations.pytorch.ToTensorV2(),
            ]),
            'valid': albumentations.Compose([
                # albumentations.random_flip(),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                albumentations.pytorch.ToTensorV2(),
            ]),
            'test': albumentations.Compose([
                # albumentations.Normalize(mean=[90, 100, 100], std=[30, 32, 28], max_pixel_value=1.0),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                albumentations.pytorch.ToTensorV2(),
            ]),
        }

    def train_dataloader(self):
        train_dataset = Custom_dataset(
            img_paths=self.train_x,
            labels=self.train_y,
            transforms=self.data_transforms['train']
        )
        return DataLoader(train_dataset, shuffle=True, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        val_dataset = Custom_dataset(
            img_paths=self.valid_x,
            labels=self.valid_y,
            transforms=self.data_transforms['valid']
        )
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def predict_dataloader(self):
        pred_dataset = Custom_dataset(
            img_paths=self.X,
            labels=None,
            transforms=self.data_transforms['test'],
            mode = 'test'
        )
        return DataLoader(pred_dataset,shuffle=False, batch_size=self.hparams.batch_size, num_workers=4)
