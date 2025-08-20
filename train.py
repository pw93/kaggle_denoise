# -*- coding: utf-8 -*-
import os
import sys
import subprocess
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


# Detect if running in Kaggle
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    '''
    !git clone https://github.com/pw93/kaggle_denoise.git /kaggle/working/code/kaggle_denoise
    !cd /kaggle/working/code/kaggle_denoise && git pull

    !git clone https://github.com/pw93/sv_denoise.git /kaggle/working/kaggle_utils
    !cd /kaggle/working/kaggle_utils && git pull

    import sys
    if '/kaggle/working/code/kaggle_denoise' not in sys.path:
        sys.path.append('/kaggle/working/code/kaggle_denoise')
    if '/kaggle/working/code/kaggle_utils' not in sys.path:
        sys.path.append('/kaggle/working/code/kaggle_utils')
    '''
    def git_clone_or_pull(repo_url, target_dir):
        if not os.path.exists(target_dir):
            subprocess.run(['git', 'clone', repo_url, target_dir], check=True)
        else:
            subprocess.run(['git', '-C', target_dir, 'pull'], check=True)

    # Clone or pull repos
    git_clone_or_pull('https://github.com/pw93/kaggle_denoise.git', '/kaggle/working/code/kaggle_denoise')
    git_clone_or_pull('https://github.com/pw93/sv_denoise.git', '/kaggle/working/kaggle_utils')

    # Add to sys.path
    paths_to_add = [
        '/kaggle/working/code/kaggle_denoise',
        '/kaggle/working/kaggle_utils',
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)






# ==============================
# Paths

if IS_KAGGLE:
    path_dataset_denoise = "/kaggle/input/denoise50"
    path_result_base = "/kaggle/working/denoise50"
else:
    path_dataset_denoise=  r"D:\data\dataset2\denoise50"
    path_result_base = r"D:\data\ai_report\denoise50-2"


train_noise = os.path.join(path_dataset_denoise, "train/input")
train_gt    = os.path.join(path_dataset_denoise, "train/gt")

val_noise = os.path.join(path_dataset_denoise, "val/input")
val_gt    = os.path.join(path_dataset_denoise, "val/gt")





num_epochs = 10
batch_size = 32
model_name = 'dncnn'

epoches_val_interval = 10
# Options
#Learning Rate Scheduler
is_use_LR_Scheduler = True
resume_from = ""  # e.g., "checkpoints/dncnn_denoise_epoch100.pth"

# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def tensor_to_image(tensor):
    """Convert torch tensor to numpy image (H, W, C) in [0, 255]"""
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

def evaluate(model, val_loader, device, criterion):
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    os.makedirs(path_result_base, exist_ok=True)
    clear_directory(path_result_base)
    dname_checkpoints = os.path.join(path_result_base, "checkpoints")
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    if model_name == "dncnn":
        model = DnCNN(image_channels=3).to(device)
    elif model_name == "dncnn10":
        model = DnCNN10(image_channels=3).to(device)
    elif model_name == "unet":
        model = UNet(n_channels=3, n_classes=3).to(device)
    else:
        raise ValueError(f'Unknown model_name: {model_name}')



    if os.path.exists(resume_from):
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


        if epoch%epoches_val_interval==1:
            avg_val_loss, avg_psnr, avg_ssim = evaluate(model, val_loader, device, criterion)

            sdata = (
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Train Loss: {avg_train_loss:.6f}  "
                    f"Val Loss: {avg_val_loss:.6f}  "
                    f"PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}  "
                    f"{timestamp}"
                )
            d = f"{epoch}\t{avg_train_loss}\t{avg_val_loss}\t{avg_psnr}\t{avg_ssim}";
            report.append(d)
        else:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            sdata = (
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Train Loss: {avg_train_loss:.6f}  "
                    f"LR: {current_lr:.6f}  "
                    f"{timestamp}"
                )
        print(sdata)





        if is_use_LR_Scheduler:
            scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            fname_checkpoint = os.path.join(dname_checkpoints, f"{model_name}_denoise_epoch{epoch}.pth")
            torch.save(model.state_dict(), fname_checkpoint)

    fname_report  = os.path.join(path_result_base, r"train_report.txt")
    write_lines(fname_report,report)


if __name__ == "__main__":
    #import torch.multiprocessing as mp
    #mp.set_start_method('spawn', force=True)
    train()
