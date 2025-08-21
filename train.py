# -*- coding: utf-8 -*-
import os


from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.unet import UNet
from model.dncnn import DnCNN
from model.dncnn import DnCNN10
from denoise_dataset_mem import DenoiseDataset_mem
from kaggle_utils.util import clear_directory, write_lines
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from kaggle_env import detect_kaggle_runtime
import platform
import multiprocessing


# ==============================
# config
is_use_kaggle = detect_kaggle_runtime()
if is_use_kaggle:
    path_dataset_denoise = "/kaggle/input/denoise50"
    path_result_base = "/kaggle/working/denoise50"
else:
    path_dataset_denoise=  r"D:\data\dataset2\noise_20_50"
    path_result_base = r"D:\data\ai_report\noise_20_50-1"

num_epochs = 1000
batch_size = 32
model_name_denoise = 'dncnn10' #'dncnn10', 'dncnn'


# advanced config
epoches_val_interval = 10
is_use_LR_Scheduler = True  #Learning Rate Scheduler
resume_from = ""  # e.g., "checkpoints/dncnn_denoise_epoch100.pth"


# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
])

def tensor_to_image(tensor):
    """Convert torch tensor to numpy image (H, W, C) in [0, 255]"""
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

def evaluate_full(model, val_loader, device, criterion):
    model.eval()
    total_val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            output2 = noisy + output  # Residual learning
            loss = criterion(output2, clean)
            total_val_loss += loss.item()

            # Convert to image for PSNR/SSIM
            for i in range(noisy.size(0)):
                pred_img = tensor_to_image(output2[i])
                clean_img = tensor_to_image(clean[i])

                psnr = compare_psnr(clean_img, pred_img, data_range=255)
                #ssim = compare_ssim(clean_img, pred_img, multichannel=True, data_range=255)
                ssim = compare_ssim(clean_img, pred_img, channel_axis=-1, data_range=255)

                total_psnr += psnr
                total_ssim += ssim
                count += 1

    avg_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    return avg_loss, avg_psnr, avg_ssim


def evaluate_basic(model, val_loader, device, criterion):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            output = model(noisy)
            output2 = noisy + output  # Residual learning
            loss = criterion(output2, clean)
            total_val_loss += loss.item()

    avg_loss = total_val_loss / len(val_loader)
    return avg_loss


def train(dname_dataset=None, danme_report=None, model_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_name == None:
        model_name = model_name_denoise

    if dname_dataset==None:
        dname_dataset_denoise = path_dataset_denoise
    else:
        dname_dataset_denoise = dname_dataset

    if danme_report==None:
        dname_result_base = path_result_base
    else:
        dname_result_base = danme_report



    train_noise = os.path.join(dname_dataset_denoise, "train/input")
    train_gt    = os.path.join(dname_dataset_denoise, "train/gt")

    val_noise = os.path.join(dname_dataset_denoise, "val/input")
    val_gt    = os.path.join(dname_dataset_denoise, "val/gt")



    os.makedirs(dname_result_base, exist_ok=True)
    clear_directory(dname_result_base)
    dname_checkpoints = os.path.join(dname_result_base, "checkpoints")
    os.makedirs(dname_checkpoints, exist_ok=True)

    # Load datasets
    train_dataset = DenoiseDataset_mem(
        noisy_dir=train_noise,
        clean_dir=train_gt,
        crop_size=(128, 128),
        transform=transform
    )
    val_dataset = DenoiseDataset_mem(
        noisy_dir=val_noise,
        clean_dir=val_gt,
        crop_size=(128, 128),
        transform=transform
    )

    def get_num_workers():
        system = platform.system()
        print(system)
        if system == 'Windows':
            return 0
        else:
            return min(4, multiprocessing.cpu_count())  # Or any logic




    n_workers = get_num_workers()
    if n_workers>0:
        persistent = True
        prefetch = 4
    else:
        persistent = False
        prefetch = None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, persistent_workers=persistent, drop_last=True, prefetch_factor=prefetch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, persistent_workers=persistent, drop_last=True, prefetch_factor=prefetch)

    # Load model
    if model_name == "dncnn":
        model = DnCNN(image_channels=3).to(device)
    elif model_name == "dncnn10":
        model = DnCNN10(image_channels=3).to(device)
    elif model_name == "unet":
        model = UNet(n_channels=3, n_classes=3).to(device)
    else:
        raise ValueError(f'Unknown model_name: {model_name}')

    #from torchsummary import summary
    #summary(model, input_size=(3, 128, 128))

    if resume_from and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from))
        print(f"Resumed from checkpoint: {resume_from}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    report=[]
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            #output: full
            #loss = criterion(output, clean)

            #output: delta or residual
            output2 = noisy + output
            loss = criterion(output2, clean)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validate
        now = datetime.now()
        timestamp = now.strftime('%H:%M:%S') + f".{int(now.microsecond / 1000):03d}"
        current_lr = optimizer.param_groups[0]['lr']
        if epoch%epoches_val_interval==1:
            avg_val_loss, avg_psnr, avg_ssim = evaluate_full(model, val_loader, device, criterion)
            sdata = (
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Train Loss: {avg_train_loss:.6f}  "
                    f"Val Loss: {avg_val_loss:.6f}  "
                    f"PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}  "
                    f"LR: {current_lr:.6f}  "
                    f"{timestamp}"
                )
            d = f"{epoch}\t{avg_train_loss}\t{avg_val_loss}\t{avg_psnr}\t{avg_ssim}";
            report.append(d)
        else:
            avg_val_loss = evaluate_basic(model, val_loader, device, criterion)
            sdata = (
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Train Loss: {avg_train_loss:.6f}  "
                    f"Val Loss: {avg_val_loss:.6f}  "
                    f"LR: {current_lr:.6f}  "
                    f"{timestamp}"
                )
        print(sdata)

        if is_use_LR_Scheduler:
            scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            fname_checkpoint = os.path.join(dname_checkpoints, f"{model_name}_denoise_epoch{epoch}.pth")
            torch.save(model.state_dict(), fname_checkpoint)

    fname_report  = os.path.join(dname_result_base, r"train_report.txt")
    write_lines(fname_report,report)


if __name__ == "__main__":
    #import torch.multiprocessing as mp
    #mp.set_start_method('spawn', force=True)
    train()
