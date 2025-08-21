# -*- coding: utf-8 -*-

import torch
from torchinfo import summary
from ptflops import get_model_complexity_info

from model.dncnn import DnCNN, DnCNN10
from model.unet import UNet

def get_model(model_name, device):
    if model_name == "dncnn":
        return DnCNN(image_channels=3).to(device)
    elif model_name == "dncnn10":
        return DnCNN10(image_channels=3).to(device)
    elif model_name == "UNet":
        return UNet(n_channels=3, n_classes=3).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def analyze_model(model_name="dncnn10", input_size=(3, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device)

    print(f"\n=== Model: {model_name} ===")
    print("\n🔎 Summary:")
    summary(model, input_size=(1, *input_size), device=str(device), verbose=2)


    print("\n📊 Computing FLOPs and Parameters:")
    with torch.cuda.device(0 if device.type == "cuda" else -1):
        macs, params = get_model_complexity_info(model, input_res=input_size, as_strings=True, print_per_layer_stat=False)
        print(f"GFLOPs : {macs}")
        print(f"Params : {params}")

if __name__ == "__main__":
    # Example usage
    #analyze_model(model_name="dncnn10", input_size=(3, 320, 480))
    analyze_model(model_name="dncnn10", input_size=(3, 3000, 1000))
