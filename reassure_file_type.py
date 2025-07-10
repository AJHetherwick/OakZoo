"""
This file double checks to make sure all file paths listed in label_df are valid.
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm


def main() -> None:

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Test_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File_Path']

    for file_path in tqdm(labeled_paths):
        
        try:
            with Image.open(file_path) as img:
                img.verify()

        except (FileNotFoundError, FileExistsError, OSError) as e:
            print(file_path, '\n', e)
        
    
main()
