"""
This file resizes images with paths in Oakland_Zoo_UW_Labels to 224x224. 
It saves images to "C:/Users/accrintern/Documents/GitHub/OakZoo/Oak_Zoo_Trail_Cam_Photos_Compressed"
"""

import pandas as pd
from PIL import Image, ImageOps
import os


def main() -> None:

    label_df_path = "C:/Users/accrintern/Documents/GitHub/OakZoo/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File_Path']

    uw_path = "Z:/Conservation/UWIN/trail cam photos/Study photos"
    nps_path = "C:/Users/accrintern/Documents/GitHub/OakZoo/NPS_Trail_Cam_Photos"

    uw_photo_paths = []
    nps_photo_paths = []

    for path in labeled_paths:  # this should be your list of full image paths
        if path.startswith(uw_path):
            uw_photo_paths.append(path)
        elif path.startswith(nps_path):
            nps_photo_paths.append(path)

        if not path.endswith('.JPG'):
            print(path)

    print('uw_photo_paths length:', len(uw_photo_paths))
    print('nps_photo_paths length:', len(nps_photo_paths))

    exit()

    # Resize urban wildlife photos from 3264x2448 to 224x224
    resize_with_padding(
    input_paths=uw_photo_paths,
    output_folder="C:/Users/accrintern/Documents/GitHub/OakZoo/resized_images",
    )

    # Resize NPS photos from irregular shapes to 224x224. Adds padding to sides if needed to preserve aspect ratio.
    resize_with_padding(
    input_paths=nps_photo_paths,
    output_folder="C:/Users/accrintern/Documents/GitHub/OakZoo/resized_images",
    )


def resize_with_padding(input_paths, output_folder, target_size=(224, 224), fill_color=(0, 0, 0)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_paths):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_paths, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    # Convert to RGB if needed
                    img = img.convert("RGB")

                    # Resize and pad
                    resized_img = ImageOps.pad(img, target_size, method=Image.BICUBIC, color=fill_color)

                    # Save
                    resized_img.save(output_path, quality=95, optimize=True)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


main()
