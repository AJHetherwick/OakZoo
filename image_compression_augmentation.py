"""
This file resizes images with paths in Oakland_Zoo_UW_Labels to 224x224. 
It then augments photos to boost class sizes to 100 per species. 
It saves images to "C:/Users/accrintern/Documents/GitHub/OakZoo/Oak_Zoo_Trail_Cam_Photos_Compressed"
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as F
import os


def main() -> None:

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)

    label_df_compressed_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv"
    label_df_compressed = pd.read_csv(label_df_compressed_path)

    # mean, std_dev = get_mean_std(label_df)

    # Get the species with under 100 observations
    freq_counts = label_df["Label_1"].value_counts(dropna=True)

    # Resize photos to 224x224
    # compress(
    #     output_folder="C:/Users/accrintern/Documents/AdamH_Project_Files/Oak_Zoo_Trail_Cam_Photos_Compressed/",
    #     freq_counts=freq_counts,
    #     label_df=label_df,
    #     label_df_compressed=label_df_compressed
    # )

    augment(label_df_compressed=label_df_compressed)


def get_mean_std(label_df: pd.DataFrame) -> tuple[list, list]:

    mean, std_dev = [], []

    for row in label_df:
        pass
        # convert to np.array and find means and stds

    return mean, std_dev


def compress(output_folder: str, freq_counts: pd.Series, label_df: pd.DataFrame, 
             label_df_compressed: pd.DataFrame, target_size=(224, 224)) -> None:
    
    # For each label in frequency counts, if over 100, randomly select 100 to compress, else compress all.
    # Save to C:/Users/accrintern/Documents/AdamH_Project_Files/Oak_Zoo_Trail_Cam_Photos_Compressed with label
    # stored in C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv

    for label, _ in freq_counts.items():

        # Create subset for species
        species_subset = label_df[label_df['Label_1'] == label]

        # Shuffle rows of species subset
        species_subset = species_subset.sample(n=min(100, len(species_subset))).reset_index(drop=True)

        # Compress either 100 or the amount of photos we have
        species_subset = species_subset.head(n=min(100, len(species_subset)))

        for _, row in species_subset.iterrows():

            file_path = row['File_Path']

            with Image.open(file_path) as img:

                if img.mode == 'RGB':
                    fill_color = (0, 0, 0)
                else:
                    fill_color = (0)

                img = ImageOps.pad(img, target_size, method=Image.BICUBIC, color=fill_color)
            
            # Save compressed photo
            filename = os.path.basename(file_path)
            full_output_path = os.path.join(output_folder, filename)
            img.save(full_output_path, quality=95, optimize=True)

            new_row = {
                'File_Path': full_output_path,
                'Label': label,
                'Label_ID': row['Label_ID_1']
            }

            label_df_compressed = pd.concat([label_df_compressed, pd.DataFrame([new_row])], ignore_index=True)
    
    label_df_compressed.to_csv('C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv', index=False)


def augment(label_df_compressed: pd.DataFrame,) -> None:

    # Augment photos for species with less than 100 observations.

    compressed_freqs = label_df_compressed['Label'].value_counts(dropna=True)
    compressed_freqs = compressed_freqs[compressed_freqs < 100]

    for label, _ in compressed_freqs.items():

        # Create subset for just under-observed species
        species_subset = label_df_compressed[label_df_compressed['Label'] == label]

        # Drop excess rows just until we reach 100 photos
        species_subset = species_subset.head(n=min(100-len(species_subset), len(species_subset)))

        for _, row in species_subset.iterrows():

            file_path = row['File_Path']

            with Image.open(file_path) as img:

                img_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor()
                ])

                aug_img = img_transform(img)

            # Save
            base, ext = os.path.splitext(file_path)

            for num in ['_aug_1', '_aug_2', '_aug_3', '_aug_4']:

                if not os.path.exists(base + num + ext):
                    file_path = base + num + ext
                    break
            aug_img = F.to_pil_image(aug_img)
            aug_img.save(file_path, quality=95, optimize=True)

            new_row = {
                'File_Path': file_path,
                'Label': label,
                'Label_ID': row['Label_ID']
            }

            # Add new row to compressed dataframe
            label_df_compressed = pd.concat([label_df_compressed, pd.DataFrame([new_row])], ignore_index=True)
    
    label_df_compressed.to_csv('C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv', index=False)



main()
