"""
This project provides species predictions for Urban Wildlife photos provided by the Oakland Zoo.
Created by Adam Hetherwick with slight guidance from Sammantha Sammons and Darren Minier.
Started 05/27/2025
"""

import pandas as pd
import numpy as np
import matplotlib
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os



def main() -> None:

    label_df_compressed_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv"
    label_df_compressed = pd.read_csv(label_df_compressed_path)
    
    tensors, labels = reformat_data(label_df_compressed)


def reformat_data(label_df_compressed) -> tuple[list, list]:

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensors = []
    labels = []

    for _, row in label_df_compressed.iterrows():
        img = Image.open(row['File_Path']).convert("RGB")
        img_tensor = transform(img)

        tensors.append(img_tensor)
        labels.append(row['Label_ID'])

    return tensors, labels


main()
