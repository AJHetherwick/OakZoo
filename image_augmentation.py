"""
This file looks augments preexisting images in Oakland_Zoo_UW_Labels for species under 100 observations
"""

import pandas as pd
from torchvision import transforms


def main() -> None:
    
    label_df_path = "C:/Users/accrintern/Documents/GitHub/OakZoo/Oakland_Zoo_UW_Labels.csv"
    label_df = pd.read_csv(label_df_path)

    freq_counts = label_df["Label_1"].value_counts(dropna=True)
    under_100_obs_spec = freq_counts[freq_counts < 100]

    for species in under_100_obs_spec:
        pass


main()
