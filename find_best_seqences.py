#%%
import os
import pandas as pd
import numpy as np

from _utils.score_utils import compute_score
'''

After running rule_based_classification with different sequences of rules,
this script analyzes the resulting score CSV files to find the best performing sequences based on F1 score.

'''
score_data_root_dir = './output/score_data_st'
file_list = os.listdir(score_data_root_dir)
csv_list = [file for file in file_list if file.endswith('.csv')]

total_f1 = {}

for idx, csv_file in enumerate(csv_list):
    file_path = os.path.join(score_data_root_dir, csv_file)
    df_score = pd.read_csv(file_path)

    score_ = compute_score(df_score)
    f1 = score_['f1']

    total_f1[csv_file] = f1

# sort by f1 score descending
sorted_f1 = dict(sorted(total_f1.items(), key=lambda item: item[1], reverse=True))
best_sequences = []

max_f1 = sorted_f1[list(sorted_f1.keys())[0]]
for file_name, f1_score in sorted_f1.items():
    if f1_score == max_f1:
        print(f"Best F1 Score File: {file_name}, F1 Score: {f1_score}")
        tmp = pd.read_csv(os.path.join(score_data_root_dir, file_name))
        sequential_ = tmp.keys().tolist()[:-2]
        best_sequences.append(sequential_)

print("Best Sequences:")
for seq in best_sequences:
    print(seq)
            


# %%
