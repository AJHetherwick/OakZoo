"""
This file looks through Oakland_Zoo_UW_Labels and returns frequency counts of each species
"""

import pandas as pd


def main() -> None:
    
    label_df_path = "C:/Users/accrintern/Documents/GitHub/OakZoo/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)

    freq_counts = label_df["Label_1"].value_counts(dropna=True)

    print(freq_counts)


main()
