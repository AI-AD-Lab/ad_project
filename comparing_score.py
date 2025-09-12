#%%
import pandas as pd
import matplotlib
import numpy as np
from pathlib import Path
import os
import shutil

# parent_dir_path = Path(__file__).resolve().parent.parent
# result_path = parent_dir_path / 'score_250703_RA_No_MAX_duration'
# best_score_path = parent_dir_path / 'best_score_no_max_duration'

base_path = Path('../FOR_REVISION')
result_path = base_path / 'score_data'
best_score_path = base_path / 'best_score_max_duration'
if not best_score_path.exists():
    best_score_path.mkdir(parents=True, exist_ok=True)

total_files = os.listdir(result_path)
score_csv_files = [score_file for score_file in total_files if score_file.endswith('.csv')]
# %%

def compute_score(conf_matrix: pd.DataFrame):
    """
    Args:
        conf_matrix: numpy 2D array (real x predicted)
    Returns:
        macro_precision, macro_recall
    """
    conf_matrix = conf_matrix.drop(columns=['TOTAL'])
    conf_matrix = conf_matrix.to_numpy()
    num_classes = conf_matrix.shape[0]

    precisions = []
    recalls = []

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    return {"precision": float(macro_precision), "recall": float(macro_recall), "f1": float(f1)}
#%%

best_f1_score = 0
file_name = ''

total_f1 = []

for file in score_csv_files:
    example = result_path / file
    pandas_result_data = pd.read_csv(example)
    score = compute_score(pandas_result_data)

    total_f1.append([score['f1'], file])

    if score['f1'] >= best_f1_score:
        best_f1_score = score['f1']
        file_name = file


# %%
sorted_data = sorted(total_f1, key=lambda x: x[0], reverse=True)
np_sort = np.array(sorted_data)
print( best_f1_score, file_name)

#%%
best_files = []
best_score_cls_priority = []
for _f1, _file in np_sort:

    if float(_f1) < best_f1_score:
        break
    print(_file)
    best_files.append(_file)
    file_path = result_path / _file.replace('.csv', '.png')
    move_to = best_score_path / _file.replace('.csv', '.png')
    shutil.copy(file_path, move_to)

    data_file_path = result_path / _file
    data = pd.read_csv(data_file_path)

    best_score_cls_priority.append(data.columns)


for_csv_data = pd.DataFrame(best_score_cls_priority)
for_csv_data.to_csv(best_score_path/'priority.csv')

#%%

for file in best_files:
    example = result_path / file
    pandas_result_data = pd.read_csv(example)
    score = compute_score(pandas_result_data)

    precision = score['precision']
    recall = score['recall']
    f1 = score['f1']

    # print('-'*30)
    # print(f"file name: {file}")
    print(f"precision:{precision}, recall:{recall}, f1-score:{f1}")

# %%

# worst case
f1_worst_case = np_sort[-1]
print(f"F1 Score - Worst Case: {f1_worst_case}")
worst_example = result_path / f1_worst_case[1]
pandas_worst_result_data = pd.read_csv(worst_example)
worst_score = compute_score(pandas_worst_result_data)
print(worst_score)

f1_worst_score = float(worst_score['f1'])
f1_best_score = float(best_score['f1'])

# normal distribution
f1_normal_distribution = np.array(np_sort[:, 0], dtype=float)

f1_mean = np.mean(f1_normal_distribution)
f1_std = np.std(f1_normal_distribution)

print(f"F1 Score - Mean: {f1_mean}, Std: {f1_std}")

import matplotlib.pyplot as plt
plt.bar(range(len(f1_normal_distribution)), f1_normal_distribution)
plt.axhline(y=f1_mean, color='r', linestyle='--', label=f'Mean ({f1_mean:.4f})')
plt.axhline(y=f1_worst_score, color='b', linestyle='--', label=f'Worst Case ({f1_worst_score:.4f})')
plt.axhline(y=f1_best_score, color='g', linestyle='--', label=f'Best Case ({f1_best_score:.4f})')
# y축 범위 제한
plt.ylim(0.5, 1)
plt.legend()
# %%
