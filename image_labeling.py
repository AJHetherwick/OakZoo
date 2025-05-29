"""
This file indexes through photos on the Oakland Zoo's staff laptop for the user to manually label.
Output is a csv file with file path, file name, label, and label ID
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def main() -> None:

    folder_path = "Z:/Conservation/UWIN/trail cam photos/Study photos/2025/January/Chabot Golf Hole 9"
    
    label_images(folder_path)


def label_images(folder_path: str) -> None:

    label_df_path = "C:/Users/accrintern/Documents/GitHub/OakZoo/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File Path']

    current_folder = os.path.normpath(folder_path)
    overlap_list = []

    for image_path in labeled_paths:
        
        image_folder = os.path.dirname(image_path)
        image_folder = os.path.normpath(image_folder)

        if image_folder == current_folder:
            overlap_list.append(image_path)
    
    remaining_images = []

    for image in os.listdir(folder_path):

        full_path = folder_path + '/' + image
        
        if full_path not in overlap_list:
            remaining_images.append(full_path)

    for full_path in remaining_images:

        # Optional: Skip non-image files
        if not full_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = mpimg.imread(full_path)

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

        plt.imshow(img)
        plt.axis("off")
        plt.title(full_path)
        plt.show(block=False)

        label_id = input('Type the integer label ID for the species in the photo. \n' \
                         'If you would like to quit with option to save enter \'break\': ').strip()

        if label_id != 'break':

            while not label_id.isdigit():
                label_id = input('Please enter a number 0-62 for label ID: ')

            reverse_label_dict = {v: k for k, v in label_label_id_dict.items()}
            label = reverse_label_dict.get(int(label_id))

            new_row = {"File Path": full_path, "Label": label, "Label ID": label_id}
            label_df = pd.concat([label_df, pd.DataFrame([new_row])], ignore_index=True)

            print(label, label_id)

            plt.close()
        
        else:
            break

    save_selection = input('Would you like to save changes to the Oak Zoo Label DF? (Enter yes or no): ').lower()

    while save_selection not in ['yes', 'no']:
        save_selection = input('Would you like to save changes to the Oak Zoo Label DF? (Enter yes or no): ').lower()
    
    if save_selection == 'yes':
        label_df.to_csv("C:/Users/accrintern/Documents/GitHub/OakZoo/Oakland_Zoo_UW_Labels.csv", index=False)


main()
