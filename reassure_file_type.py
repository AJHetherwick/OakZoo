"""
This file resizes images with paths in Oakland_Zoo_UW_Labels to 224x224. 
It saves images to "C:/Users/accrintern/Documents/AdamH_Project_Files/Oak_Zoo_Trail_Cam_Photos_Compressed"
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm


def main() -> None:

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File_Path']

    for file_path in tqdm(labeled_paths):
        
        try:
            with Image.open(file_path) as img:
                img.verify()

        except (FileNotFoundError, FileExistsError, OSError):
            print(file_path)
        
    
main()
