"""
This file looks through Oakland_Zoo_UW_Labels and returns frequency counts of each species
"""

import pandas as pd


def main() -> None:
    
    test_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Test_Labels_Compressed.csv"
    test_df = pd.read_csv(test_df_path)

    train_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_Train_Labels_Compressed.csv"
    train_df = pd.read_csv(train_df_path)

    combined_df = pd.concat([test_df, train_df])

    freq_counts = combined_df["Label"].value_counts(dropna=True)

    print(freq_counts)


main()
