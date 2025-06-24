"""
This file indexes through photos on the Oakland Zoo's staff laptop for the user to manually label.
Output is a csv file with file path, file name, label, and label ID
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def main() -> None:

    folder_path = "Z:/Conservation/UWIN/trail cam photos/Study photos/2024/July 2024/Creek"
    
    label_images(folder_path)


def label_images(folder_path: str) -> None:

    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)
    labeled_paths = label_df['File_Path']

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

    for full_path in remaining_images:

        # Optional: Skip non-image files
        if not full_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = mpimg.imread(full_path)

        plt.imshow(img)
        plt.axis("off")
        plt.title(full_path)
        plt.show(block=False)

        # Get user labels. If user enters 'break', break for loop.

        user_break, label_df = get_label(label_df, reverse_label_dict, full_path)
        if not user_break:
            break

    save_selection = input('Would you like to save changes to the Oak Zoo Label DF? (Enter yes or no): ').lower()

    while save_selection not in ['yes', 'no']:
        save_selection = input('Would you like to save changes to the Oak Zoo Label DF? (Enter yes or no): ').lower()
    
    if save_selection == 'yes':
        label_df.to_csv("C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels.csv", index=False, na_rep="N/A")


def get_label(label_df: pd.DataFrame, reverse_label_dict: dict, full_path: str) -> tuple[bool, pd.DataFrame]:

    label_1_data = user_label_selection(reverse_label_dict)

    if label_1_data[0]:

        if input('If there is more than one label enter any key here:'):

            label_2_data = user_label_selection(reverse_label_dict)
        
        else:
            label_2_data = True, ['N/A', 'N/A', 'N/A']
    
    else:
        label_2_data = True, ['N/A', 'N/A', 'N/A']

    if label_1_data[0] and label_2_data[0]:

        new_row = {"File_Path": full_path, 
                   "Label_1": label_1_data[1][0], 
                   "Label_ID_1": label_1_data[1][1], 
                   "Quantity_1": label_1_data[1][2],
                   "Label_2": label_2_data[1][0],
                   "Label_ID_2": label_2_data[1][1],
                   "Quantity_2": label_2_data[1][2]}
        
        label_df = pd.concat([label_df, pd.DataFrame([new_row])], ignore_index=True)

        print(new_row)

    else:
        return False, label_df

    plt.close()

    return True, label_df


def user_label_selection(reverse_label_dict: dict) -> tuple[bool, list]:

    label_id = input('Type the integer label ID for the species in the photo. \n' \
                         'If you would like to quit with option to save enter \'break\': ').strip()

    if label_id != 'break':

        label_id = ensure_digit('label_id', label_id)

        quantity = get_quantity(label_id)    

        label = reverse_label_dict.get(int(label_id))

        add_label = True
    
    else:
        return False, ['', '', '']
    
    return add_label, [label, label_id, quantity]


def get_quantity(label_id: int) -> int:

    quantity = input('Enter quantity if more than 1:')

    if label_id in [26, 55, 58]:
        quantity = 'N/A'
    elif quantity:
        quantity = ensure_digit('quantity', quantity)
    else:
        quantity = 1

    return quantity


def ensure_digit(col: str, val) -> int:
    # Ensure label_id or quanitity input is a digit

    while not val.isdigit():
        val = input(f'Please enter a digit for {col}: ')

    while int(val) > 64:
        val = input(f'Please enter a digit for {col}: ')
    
    return int(val)


main()
