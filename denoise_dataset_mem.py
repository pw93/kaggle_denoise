# -*- coding: utf-8 -*-

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


#return : A tuple: (noisy_tensor, clean_tensor)
#With pixel values in [0, 1] range (ToTensor() does this)
class DenoiseDataset_mem(Dataset):
    def __init__(self, noisy_dir, clean_dir, crop_size=(128, 128), transform=None):
        """
        Args:
            noisy_dir (str): Path to the directory with noisy images.
            clean_dir (str): Path to the directory with clean images.
            crop_size (tuple): Desired output size of the crop (height, width).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.crop_size = crop_size
        self.transform = transform

        # Ensure both directories contain the same number of images
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        assert len(self.noisy_images) == len(self.clean_images), \
            "Noisy and clean directories must have the same number of images."

        self.noisy_data=[]
        self.clean_data=[]
        sz = len(self.noisy_images)
        for idx in range(sz):
            noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
            clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
            noisy_img = Image.open(noisy_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
            self.noisy_data.append(noisy_img)
            self.clean_data.append(clean_img)




    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):

        noisy_img = self.noisy_data[idx]
        clean_img = self.clean_data[idx]

        # Apply random crop
        i, j, h, w = transforms.RandomCrop.get_params(noisy_img, self.crop_size)
        noisy_img = transforms.functional.crop(noisy_img, i, j, h, w)
        clean_img = transforms.functional.crop(clean_img, i, j, h, w)

        # Apply transformations
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img
