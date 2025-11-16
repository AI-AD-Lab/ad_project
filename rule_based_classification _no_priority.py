#%% IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from config import config

from _utils.data_processing_utils import data_load
from _utils.plot_utils import plot_confusion_matrix_table

from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

# CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / "simulation_TOTAL_250626_2"
# SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
labels = ['RA', 'UT', 'LT', 'RT', 'ST', 'LLC', 'RLC']

# MAIN CLASSIFICATION FUNCTION
def excute_rule_based_classification_no_priority(class_perm:list[str]) -> pd.DataFrame:
    labeled_data = [[] for _ in range(len(labels))]
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SYN_LOG_DATA_ROOT_DIR / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        COUNT = 1

        LLC = detect_left_lane_change(data, duration_sec=0.875357, threshold=0.212362)
        RLC = detect_right_lane_change(data, duration_sec=0.833088, threshold=0.137831)
        ST = detect_straight(data, abs_normal_threshold=0.059045, abs_threshold=0.109141, duration_sec=7.532441)
        RT = detect_right_turn(data, right_threshold=1.183795, duration_sec=1.574704,  max_duration_sec=5.47996)
        LT = detect_left_turn(data, left_threshold=-0.427932, duration_sec=4.137019, max_duration_sec=6.156220)
        UT = detect_u_turn(data, threshold=2.187467, duration_sec=2.823866)
        RA = detect_roundabout(data, threshold_neg=-0.146373, threshold_pos=0.097355, duration_sec=1.454986, max_duration_sec=14.449369)

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
