#%% IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from config import config

from _utils.data_processing_utils import rolling, data_load
from _utils.plot_utils import plot_confusion_matrix_table

from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

# CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']

# MAIN CLASSIFICATION FUNCTION
def excute_rule_based_classification_no_priority(class_perm:list[str]) -> pd.DataFrame:
    labeled_data = [[] for _ in range(len(labels))]
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SYN_LOG_DATA_ROOT_DIR / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        COUNT = 1

        LLC = detect_left_lane_change(data, duration_sec=0.7, threshold=0.2)
        RLC = detect_right_lane_change(data, duration_sec=0.7, threshold=0.2)
        ST = detect_straight(data, abs_normal_threshold=0.05, abs_threshold=0.3, duration_sec=8)
        RT = detect_right_turn(data)
        LT = detect_left_turn(data)
        RA = detect_roundabout(data, max_duration_sec=15)
        UT = detect_u_turn(data)

        label_variable = {
            'RA': RA, 'ST': ST, 'UT': UT,
            'LT': LT, 'RT': RT, 'LLC': LLC, 'RLC': RLC
        }

        values = [label_variable[label] for label in class_perm]

        result_list = [0] * 9
        for i, value in enumerate(values):
            if value:
                result_list[i] = value

        if not any(values):
            result_list[-2] = 1  # NO_LABEL

        result_list[-1] = COUNT
        labeled_data[real_index[label]].append(result_list)

    total_result = []
    for i, label in enumerate(labeled_data):
        np_sliced = np.array(label)
        np_sliced_change = [[0 if x is None else x for x in row] for row in np_sliced]
        column_sum = np.sum(np_sliced_change, axis=0)
        total_result.append(column_sum)

    perm_array = list(class_perm)
    df_total_result = pd.DataFrame(total_result, columns=perm_array + ["NO_LABEL", "TOTAL"], index=perm_array)
    return df_total_result

# MAIN RUN
if __name__ == "__main__":
    df_total_result = excute_rule_based_classification_no_priority(class_perm=labels)

    save_dir = Path('./output/plots/score')
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_dir / f"no_priority.png"
    csv_path = save_dir / f"no_priority.csv"

    plot_confusion_matrix_table(df_total_result, save_path=str(plot_path))
    df_total_result.to_csv(csv_path, index=False)
