#%% IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from _utils.utils_plot import time_base_plot, draw_ay_plot
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from config import config
import itertools

from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

#%% CONFIG
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH / 'simulation_TOTAL_250626'

cls_label = config['class_to_label']
label_cls = config['label_to_class']
short_to_long_label = config['Short_to_Long_Label']

label_data = pd.read_csv(SINGLE_SCENARIO_SYNLOG_DATA_ROOT / 'label.csv')

# RA ST UT LT RT LLC RLC
labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']

#%% DATA LOAD
def data_load(data_file_path):
    USEDCOLUMNS = config['data_columns']
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")
    _data = pd.read_csv(data_file_path)
    _data = _data[USEDCOLUMNS]
    return _data

def pandas_plot_save(df, save_path:None|str=None):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.scale(1, 1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

#%% MAIN CLASSIFICATION FUNCTION
def excute_rule_based_classification_no_priority(class_perm:list[str]) -> pd.DataFrame:
    labeled_data = [[] for _ in range(len(cls_label))]
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        COUNT = 1

        LLC = detect_left_lane_change(data, duration_sec=0.7, threshold=0.2)
        RLC = detect_right_lane_change(data, duration_sec=0.7, threshold=0.2)
        ST = detect_straight(data, abs_normal_threshold=0.05, abs_threshold=0.3, duration_sec=8)
        RT = detect_right_turn(data)
        LT = detect_left_turn(data)
        RA = detect_roundabout(data)
        UT = detect_u_turn(data)

        label_variable = {
            'RA': RA, 'ST': ST, 'UT': UT,
            'LT': LT, 'RT': RT, 'LLC': LLC, 'RLC': RLC
        }

        values = [label_variable[label] for label in class_perm]

        if not any(values):
            result_list = [0] * 9
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

def process_one_perm():
    df_total_result = excute_rule_based_classification_no_priority(class_perm=labels)

    save_dir = Path('./output/plots/score')
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_dir / f"no_priority.png"
    csv_path = save_dir / f"no_priority.csv"

    pandas_plot_save(df_total_result, save_path=str(plot_path))
    df_total_result.to_csv(csv_path, index=False)


#%% MAIN RUN
if __name__ == "__main__":
    process_one_perm()