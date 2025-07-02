#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from _utils.data_processing_utils import normalize_time
from _utils.utils_plot import time_base_plot, draw_ay_plot
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from config import config
import itertools
#%%
from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

#%%


# SAMPLING_RATE = config['sampling_rate']

def data_load(data_file_path):
    """ Load data from a CSV file and normalize the time column. """
    USEDCOLUMNS = config['data_columns']

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")

    # Load the data
    _data = pd.read_csv(data_file_path)
    _data = _data[USEDCOLUMNS]
    return _data

def pandas_plot_save(df, save_path:None|str=None):
    ''' Render a pandas DataFrame as a table and save or show it. '''

    fig, ax = plt.subplots(figsize=(5, 2))

    # 테이블 렌더링
    ax.axis('off')  # 축 제거
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')

    table.scale(1, 1.5)  # 크기 조절
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)

    plt.close(fig)


# def excute_rule_based_classification(perm:list[str]) -> None:
#     """
#     Execute the rule-based classification for trajectory types.
#     This function processes the data, applies classification rules, and saves the results.
#     """

#     labeled_data = [[] for _ in range(len(cls_label))]
#     real_index = { short_to_long_label[label]:idx for idx, label in enumerate(perm)}

#     pass  # Placeholder for the main execution logic if needed


# 기본 경로 설정
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH /  'simulation_TOTAL_250626'  # 시간이 일정한 데이터 파일: SYNC

cls_label = config['class_to_label']
label_cls = config['label_to_class']
short_to_long_label = config['Short_to_Long_Label']

label_data = pd.read_csv(SINGLE_SCENARIO_SYNLOG_DATA_ROOT / 'label.csv')
labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']
perms = list(itertools.permutations(labels))

for perm_idx, perm in enumerate(perms):
    labeled_data = [[] for _ in range(len(cls_label))]
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(perm)}

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):

        file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        NO_LABEL = 0
        COUNT = 1

        # predict each label
        LLC = detect_left_lane_change(data, duration_sec=0.7, threshold=0.2)
        RLC = detect_right_lane_change(data, duration_sec=0.7, threshold=0.2)
        ST = detect_straight(data, abs_normal_threshold=0.05, abs_threshold=0.3, duration_sec=8)
        RT = detect_right_turn(data)
        LT = detect_left_turn(data)
        RA = detect_roundabout(data)
        UT = detect_u_turn(data)

        label_variable = { # update label_variable to match the perm order
            'RA': RA,
            'ST': ST,
            'UT': UT,
            'LT': LT,
            'RT': RT,
            'LLC': LLC,
            'RLC': RLC
        }

        values = [ label_variable[label] for label in perm ] # 우선순위에 맞게 정렬, prediction

        # 첫 번째만 셀렉
        for i, value in enumerate(values) :
            if value:
                result_list = [0] * 9
                result_list[i] = 1
                break  # 첫 번째 1만 인정

        # No label case
        if not any(values):
            result_list = [0] * 9
            result_list[-2] = 1

        result_list[-1] = COUNT
        labeled_data[real_index[label]].append(result_list) # real-> prediction 넣기

    total_result = []
    for i, label in enumerate(labeled_data):
        np_sliced = np.array(label)
        np_sliced_change = [[0 if x is None else x for x in row] for row in np_sliced]
        column_sum = np.sum(np_sliced_change, axis=0)
        total_result.append(column_sum)

    perm_array = list(perm)
    df_total_result = pd.DataFrame(total_result, columns=perm_array + ["NO_LABEL", "TOTAL"], index=perm_array)

    pandas_save_path = './output/plots/score/'
    if not os.path.exists(pandas_save_path):
        os.makedirs(pandas_save_path)
    pandas_plot_save(df_total_result, save_path=pandas_save_path + f"total_result_{perm_idx}.png")
    df_total_result.to_csv(pandas_save_path + f"total_result_{perm_idx}.csv", index=False)

    if perm_idx % 100 == 0:
        print(f'Processed {perm_idx} permutations...')

