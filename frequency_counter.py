"""
This file looks through Oakland_Zoo_UW_Labels and returns frequency counts of each species
"""

import pandas as pd


def main() -> None:
    
    label_df_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)

    freq_counts = label_df["Label_1"].value_counts(dropna=True)

    print(freq_counts)

    labeled_df_compressed_path = "C:/Users/accrintern/Documents/AdamH_Project_Files/Oakland_Zoo_UW_Labels_Compressed.csv"
    labeled_df_compressed = pd.read_csv(labeled_df_compressed_path)

    freq_counts_compressed = labeled_df_compressed['Label'].value_counts(dropna=True)

    print(freq_counts_compressed)


main()
