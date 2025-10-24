import os
import pandas as pd
import numpy as np

import _utils.score_utils as score_utils

data_path_1 = "output\plots\score\confusion_matrix2.csv"
data_path_2 = "output\plots\score\confusion_matrix.csv"

data_1 = pd.read_csv(data_path_1)
data_2 = pd.read_csv(data_path_2)

score_1 = score_utils.compute_score(data_1)
score_2 = score_utils.compute_score(data_2)

print("=== Confusion Matrix 1 ===")
for key, value in score_1.items():
    print(f"{key}: {value:.4f}")
print("\n=== Confusion Matrix 2 ===")
for key, value in score_2.items():
    print(f"{key}: {value:.4f}")
    