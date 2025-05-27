"""
This file indexes through photos on the Oakland Zoo's staff laptop for the user to manually label.
Output is a csv file with file path, file name, label, and label ID
"""

import pandas as pd
from PIL import Image


def main() -> None:

    folder_path = "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/January/Chabot Golf Hole 9"
    
    label_images(folder_path)


def label_images(folder_path: str) -> None:

    label_df_path = "C:\Users\accrintern\Documents\GitHub\OakZoo\Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    
    for image in folder_path:

        Image.open(image).convert("RGB")

        label = input('Enter the label for this photo')
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
            'Deer Mouse': 20,
            'Domestic Cat': 21,
            'Domestic Dog': 22,
            'Domestic Horse': 23,
            'Empty': 24,
            'European Starling': 25,
            'Fish': 26,
            'Fox (cannot ID)': 27,
            'Fox Squirrel': 28,
            'Gray Fox': 29,
            'Great Blue Heron': 30,
            'Great Egret': 31,
            'Hawk (cannot ID)': 32,
            'Human': 33,
            'Hummingbird': 34,
            'Insect (cannot ID)': 35,
            'Lizard (cannot ID)': 36,
            'Mallard': 37,
            'Mourning Dove': 38,
            'North American River Otter': 39,
            'Owl': 40,
            'Quail': 41,
            'Rabbit (cannot ID)': 42,
            'Racoon': 43,
            'Ring-Necked Duck': 44,
            'small Mammal (cannot ID)': 45,
            'Snowy Egret': 46,
            'Spotted Towhee': 47,
            'Steller\'s Jay': 48,
            'Striped Skunk': 49,
            'Swallow (cannot ID)': 50,
            'Tamaulipas Crow': 51,
            'Turkey Vulture': 52,
            'Unknown': 53,
            'Vehicle': 54,
            'Virginia Opossum': 55,
            'WATER': 56,
            'Western Gray Squirrel': 57,
            'White Crowned Sparrow': 58,
            'Wild Pig': 59,
            'Wild Turkey': 60,
            'Willet': 61,
            'Woodrat': 62
        }

        label_id = input('Type the integer label ID for the species in the photo')
        label = next((k for k, v in label_label_id_dict.items() if v == label_id), None)

        new_row = {folder_path + image, label, label_id}

        label_df = pd.concat([label_df, pd.DataFrame([new_row])], ignore_index=True)



