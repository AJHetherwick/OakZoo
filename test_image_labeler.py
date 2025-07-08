"""
This file indexes through 2025 April photos on the Oakland Zoo's staff laptop for the user to manually label.
Output is a csv file with file path, file name, label, and label ID.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random


def main() -> None:

    folder_paths = ["Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/EBRP Goldenrod",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/EBRP Honker Bay",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/EBRP Horseshoe",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/EBRP MacDonald Trail",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/Keyes Residence",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/LCG Hole 9",
                    "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/April/LCG Hole 18"]
    
    label_images(folder_paths)


def label_images(folder_paths: list[str]) -> None:

    label_label_id_dict = {
            'Acorn Woodpecker': 0,
            'American Coot': 1,
            'American Robin': 2,
            'Band-tailed Pigeon': 3,
            'Bat Ray': 4,
            'Bird': 5,
            'Black-Crowned Night Heron': 6,
            'Black-tailed Deer': 7,
            'Black-tailed Jackrabbit': 8,
            'Blue Mussel': 9,
            'Bobcat': 10,
            'Brown Pelican': 11,
            'Brush Rabbit': 12,
            'California Ground Squirrel': 13,
            'California Gull': 14,
            'California Scrub Jay': 15,
            'California Thrasher': 16,
            'Canada Goose': 17,
            'Chirping Sparrow': 18,
            'Common Goldeneye': 19,
            'Coyote': 20,
            'Deer Mouse': 21,
            'Domestic Cat': 22,
            'Domestic Dog': 23,
            'Domestic Goat': 24,
            'Domestic Horse': 25,
            'Empty': 26,
            'European Starling': 27,
            'Fish': 28,
            'Fox (cannot ID)': 29,
            'Fox Squirrel': 30,
            'Gray Fox': 31,
            'Great Blue Heron': 32,
            'Great Egret': 33,
            'Hawk (cannot ID)': 34,
            'Human': 35,
            'Hummingbird': 36,
            'Insect (cannot ID)': 37,
            'Lizard (cannot ID)': 38,
            'Mallard': 39,
            'Mourning Dove': 40,
            'North American River Otter': 41,
            'Owl': 42,
            'Quail': 43,
            'Rabbit (cannot ID)': 44,
            'Raccoon': 45,
            'Ring-Necked Duck': 46,
            'small Mammal (cannot ID)': 47,
            'Snowy Egret': 48,
            'Spotted Towhee': 49,
            'Steller\'s Jay': 50,
            'Striped Skunk': 51,
            'Swallow (cannot ID)': 52,
            'Tamaulipas Crow': 53,
            'Turkey Vulture': 54,
            'Unknown': 55,
            'Vehicle': 56,
            'Virginia Opossum': 57,
            'WATER': 58,
            'Western Gray Squirrel': 59,
            'White Crowned Sparrow': 60,
            'Wild Pig': 61,
            'Wild Turkey': 62,
            'Willet': 63,
            'Woodrat': 64
        }
    
    reverse_label_dict = {v: k for k, v in label_label_id_dict.items()}

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Test_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File_Path']
    labeled_paths = [os.path.normpath(p) for p in labeled_paths]

    full_images_list = []

    for folder in folder_paths:

        for image in os.listdir(folder):

            full_path = folder + '/' + image

            full_images_list.append(full_path)

    random.shuffle(full_images_list)    # Shuffle list so I am less likely to review the same photos over again.

    for image_path in full_images_list:

        if os.path.normpath(image_path) in labeled_paths:
            continue
            
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = mpimg.imread(image_path)

        plt.imshow(img)
        plt.axis("off")
        plt.title(image_path)
        plt.show(block=False)

        species_label_id = get_label()
            
        if species_label_id == 'break':
            break
        elif species_label_id == '':
            continue
        else:
            species_label_id = int(species_label_id)

            new_row = {"File_Path": image_path,
                           "Label": reverse_label_dict[species_label_id],
                           "Label_ID": species_label_id}
            
            label_df = pd.concat([label_df, pd.DataFrame([new_row])], ignore_index=True)
            
        plt.close()

        if species_label_id == 'break':
            break
    
    if input('If you would like to save changes enter \'y\':').lower() == 'y':
        label_df.to_csv(label_df_path, index=False)

    
def get_label() -> str:

    species_label_id = input('Enter the label id for species in photo\n' \
                             'Enter \'break\' to save and quit:')

    return species_label_id


main()
