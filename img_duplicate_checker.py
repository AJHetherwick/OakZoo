"""This notebook checks to find duplicate images in training and test set."""


from PIL import Image
import hashlib
import pandas as pd


def main() -> None:

    test_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Test_Labels_Compressed.csv"
    test_df = pd.read_csv(test_df_path)

    train_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Train_Labels_Compressed.csv"
    train_df = pd.read_csv(train_df_path)

    train_paths = train_df['File_Path']
    test_paths = test_df['File_Path']

    train_hashes = {hash_image(p): p for p in train_paths}
    test_hashes = {hash_image(p): p for p in test_paths}

    duplicates = set(train_hashes) & set(test_hashes)

    for h in duplicates:
        print("test:", test_hashes[h])
        print("train:", train_hashes[h])


def hash_image(path) -> None:
    with Image.open(path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()


main()
