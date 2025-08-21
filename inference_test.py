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


path_dataset_denoise=  r"D:\data\dataset2\denoise50"
result_dir = 'D:\\data\\ai_report\\denoise50-4\\inf_test'
checkpoint_path = r'D:\data\ai_report\denoise50-4\checkpoints\dncnn_denoise_epoch680.pth'
num_samples = 20
model_name = "dncnn10"

#=====================================
#=====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(result_dir, exist_ok=True)
clear_directory(result_dir)
test_dir = os.path.join(path_dataset_denoise,"test\input")
gt_dir = os.path.join(path_dataset_denoise,"test\gt")

# === Load Model ===
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

def write_image(np_img, path):
    img = Image.fromarray((np_img * 255).astype(np.uint8))
    img.save(path)


image_paths = sorted(glob(os.path.join(test_dir, '*.png')))
if num_samples>0:
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

    shutil.copy2(path, os.path.join(result_dir, f'{name}_noise.png'))
    write_image(denoised_img, os.path.join(result_dir, f'{name}_ai.png'))

print(f"Denoised results saved to: {result_dir}")
