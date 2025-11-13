import torch 
import torch.nn as nn
import os
import random
from PIL import Image
import torchvision.transforms.functional as TF
from config import patch_size, scale, dir_HR, dir_LR

class SRDataset(torch.utils.data.Dataset):
    def __init__(self, dir_HR, dir_LR, scale = scale, patch_size = patch_size, max_samples = None):
        super().__init__()

        ## read all the paths

        self.dir_HR = dir_HR
        self.dir_LR = dir_LR
        self.filenames = sorted(os.listdir(self.dir_HR))
        # Limit to max_samples if specified
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        hr_path = os.path.join(self.dir_HR, self.filenames[idx])
        lr_name = self.filenames[idx].replace('.png', f'x{self.scale}.png')
        lr_path = os.path.join(self.dir_LR, lr_name)
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # ------------------------------
        # Random crop, aligned in HR/LR
        # ------------------------------
        w_lr, h_lr = lr.size
        p_lr = self.patch_size // self.scale

        if w_lr > p_lr and h_lr > p_lr:
            x_lr = random.randint(0, w_lr - p_lr)
            y_lr = random.randint(0, h_lr - p_lr)
        else:
            x_lr, y_lr = 0, 0  # small images fallback

        # crop LR
        lr_crop = lr.crop((x_lr, y_lr, x_lr + p_lr, y_lr + p_lr))
        # corresponding HR crop
        hr_crop = hr.crop((
            x_lr * self.scale,
            y_lr * self.scale,
            (x_lr + p_lr) * self.scale,
            (y_lr + p_lr) * self.scale
        ))

        # ------------------------------
        # Random flip/rotate augmentations
        # ------------------------------
        if random.random() < 0.5:
            lr_crop = TF.hflip(lr_crop)
            hr_crop = TF.hflip(hr_crop)
        if random.random() < 0.5:
            lr_crop = TF.vflip(lr_crop)
            hr_crop = TF.vflip(hr_crop)
        if random.random() < 0.5:
            lr_crop = lr_crop.rotate(180)
            hr_crop = hr_crop.rotate(180)

        # ------------------------------
        # Convert to tensors [0,1]
        # ------------------------------
        lr_t = TF.to_tensor(lr_crop)
        hr_t = TF.to_tensor(hr_crop)

        return hr_t, lr_t

train_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size
)

# Mini dataset with 8 images for testing
mini_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size,
    max_samples=8
)