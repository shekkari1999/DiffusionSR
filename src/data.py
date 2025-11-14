import torch 
import torch.nn as nn
import os
import random
from PIL import Image
import torchvision.transforms.functional as TF
from config import patch_size, scale, dir_HR, dir_LR, use_patches

class SRDataset(torch.utils.data.Dataset):
    def __init__(self, dir_HR, dir_LR, scale = scale, patch_size = patch_size, max_samples = None, use_patches = False):
        super().__init__()

        self.dir_HR = dir_HR
        self.dir_LR = dir_LR
        self.scale = scale
        self.patch_size = patch_size
        self.use_patches = use_patches  # If True, expects pre-generated patches (both 256x256)
        
        if use_patches:
            # For pre-generated patches: HR and LR have same filenames, both are 256x256
            self.filenames = sorted([f for f in os.listdir(self.dir_HR) if f.endswith('.png')])
        else:
            # For full images: LR filenames have 'x4' suffix
            self.filenames = sorted(os.listdir(self.dir_HR))
        
        # Limit to max_samples if specified
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.dir_HR, self.filenames[idx])
        
        if self.use_patches:
            # Pre-generated patches: same filename for both HR and LR (both 256x256)
            lr_path = os.path.join(self.dir_LR, self.filenames[idx])
            hr = Image.open(hr_path).convert("RGB")
            lr = Image.open(lr_path).convert("RGB")
            
            # Both are already 256x256 patches, no cropping needed
            # Just apply augmentations
        else:
            # Full images: need to crop patches
            lr_name = self.filenames[idx].replace('.png', f'x{self.scale}.png')
            lr_path = os.path.join(self.dir_LR, lr_name)
            hr = Image.open(hr_path).convert("RGB")
            lr = Image.open(lr_path).convert("RGB")
            
            # Random crop, aligned in HR/LR
            w_lr, h_lr = lr.size
            p_lr = self.patch_size // self.scale

            if w_lr > p_lr and h_lr > p_lr:
                x_lr = random.randint(0, w_lr - p_lr)
                y_lr = random.randint(0, h_lr - p_lr)
            else:
                x_lr, y_lr = 0, 0  # small images fallback

            # crop LR
            lr = lr.crop((x_lr, y_lr, x_lr + p_lr, y_lr + p_lr))
            # corresponding HR crop
            hr = hr.crop((
                x_lr * self.scale,
                y_lr * self.scale,
                (x_lr + p_lr) * self.scale,
                (y_lr + p_lr) * self.scale
            ))

        # ------------------------------
        # Random flip/rotate augmentations
        # ------------------------------
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
        if random.random() < 0.5:
            lr = lr.rotate(180)
            hr = hr.rotate(180)

        # ------------------------------
        # Convert to tensors [0,1]
        # ------------------------------
        lr_t = TF.to_tensor(lr)
        hr_t = TF.to_tensor(hr)

        return hr_t, lr_t

train_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size,
    use_patches=use_patches
)

# Mini dataset with 8 images for testing
mini_dataset = SRDataset(
    dir_HR=dir_HR,
    dir_LR=dir_LR,
    scale=scale,
    patch_size=patch_size,
    max_samples=8,
    use_patches=use_patches
)

print(f"Full training dataset size: {len(train_dataset)}")
print(f"Mini dataset size: {len(mini_dataset)}")
