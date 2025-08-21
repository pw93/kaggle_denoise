# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from glob import glob
from model.dncnn import DnCNN
from model.dncnn import DnCNN10
from model.unet import UNet
from utils2.util import clear_directory
import shutil

#=====================================
# config



test_dir = r"D:\data\dataset2\noise_20_50\test\input"
gt_dir = r"D:\data\dataset2\noise_20_50\test\gt"
result_dir = 'c:\\temp3\\denoise_mix-5'
checkpoint_path = r'checkpoints\dncnn10_denoise_epoch1000.pth'


num_samples = 20
model_name = "dncnn10"

#=====================================
os.makedirs(result_dir, exist_ok=True)
clear_directory(result_dir)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name=="dncnn":
    model = DnCNN(image_channels=3).to(device)
elif model_name=="dncnn10":
    model = DnCNN10(image_channels=3).to(device)
elif model_name=="UNet":
    model = UNet(n_channels=3, n_classes=3).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()




# === Image Transform ===
transform = transforms.Compose([
    #transforms.CenterCrop(patch_size),
    transforms.ToTensor()
])

def to_image(tensor):
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)

def save_image(np_img, path):
    img = Image.fromarray((np_img * 255).astype(np.uint8))
    img.save(path)





# === Load 10 test images ===
image_paths = sorted(glob(os.path.join(test_dir, '*.png')))
image_paths = image_paths[:num_samples]

# === Process and Save ===
for path in image_paths:
    name = os.path.splitext(os.path.basename(path))[0]  # e.g., "0001"

    noisy_img = Image.open(path).convert('RGB')
    noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)




    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)

        output2 = noisy_tensor + denoised_tensor



        denoised_tensor = torch.clamp(output2, 0., 1.)

    denoised_img = to_image(denoised_tensor)


    # Save images
    if gt_dir is not None and gt_dir != "":
        gt_path = os.path.join(gt_dir, f"{name}.png")
        if os.path.exists(gt_path):
            shutil.copy2(gt_path, os.path.join(result_dir, f'{name}_gt.png'))

    noisy_img.save(os.path.join(result_dir, f'{name}_noise.png'))
    save_image(denoised_img, os.path.join(result_dir, f'{name}_ai.png'))

print(f"Denoised results saved to: {result_dir}")
