# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from glob import glob
import shutil
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from model.dncnn import DnCNN
from model.dncnn import DnCNN10
from model.unet import UNet
from utils2.util import clear_directory
#===============================
# config
path_dataset_denoise=r"D:\data\dataset2\denoise50"
result_dir     =r'D:\data\ai_report\noise_20_50-1\inf_test50'
#checkpoint_path=r'D:\data\ai_report\denoise20-1\checkpoints\dncnn10_denoise_epoch900.pth'
checkpoint_path=r'D:\data\ai_report\noise_20_50-1\checkpoints\dncnn10_denoise_epoch900.pth'


model_name="dncnn10"
num_samples=0
#===============================

def inf_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(result_dir, exist_ok=True)
    clear_directory(result_dir)

    test_dir = os.path.join(path_dataset_denoise, "test/input")
    gt_dir = os.path.join(path_dataset_denoise, "test/gt")

    # === Load Model ===
    if model_name == "dncnn":
        model = DnCNN(image_channels=3).to(device)
    elif model_name == "dncnn10":
        model = DnCNN10(image_channels=3).to(device)
    elif model_name == "UNet":
        model = UNet(n_channels=3, n_classes=3).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    def to_image(tensor):
        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        return np.clip(img, 0, 1)

    def write_image(np_img, path):
        img = Image.fromarray((np_img * 255).astype(np.uint8))
        img.save(path)

    def compute_metrics(img1, img2):
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)
        ssim_val = ssim_metric(img1, img2, channel_axis=2, data_range=255)
        psnr_val = psnr_metric(img1, img2, data_range=255)
        return ssim_val, psnr_val

    image_paths = sorted(glob(os.path.join(test_dir, '*.png')))
    if num_samples > 0:
        image_paths = image_paths[:num_samples]

    report_path = os.path.join(result_dir, 'report_inf_test.txt')
    with open(report_path, 'w') as report_file:
        report_file.write("Filename\tSSIM\tPSNR\n")

        for path in image_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            noisy_img = Image.open(path).convert('RGB')
            noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)

            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)
                output2 = noisy_tensor + denoised_tensor
                denoised_tensor = torch.clamp(output2, 0., 1.)

            denoised_img = to_image(denoised_tensor)
            write_image(denoised_img, os.path.join(result_dir, f'{name}_ai.png'))
            shutil.copy2(path, os.path.join(result_dir, f'{name}_noise.png'))

            if gt_dir and os.path.exists(os.path.join(gt_dir, f"{name}.png")):
                gt_img = Image.open(os.path.join(gt_dir, f"{name}.png")).convert('RGB')
                gt_tensor = transform(gt_img).unsqueeze(0).to(device)
                gt_img_np = to_image(gt_tensor)

                shutil.copy2(os.path.join(gt_dir, f"{name}.png"), os.path.join(result_dir, f'{name}_gt.png'))

                ssim_val, psnr_val = compute_metrics(denoised_img, gt_img_np)
                report_file.write(f"{name}\t{ssim_val:.4f}\t{psnr_val:.2f}\n")
            else:
                report_file.write(f"{name}\tN/A\tN/A\n")

    print(f"Denoised results and report saved to: {result_dir}")

# If running directly
if __name__ == "__main__":
    inf_test()
