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

#%%
from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

#%%

COLUMN_NAMES = config['data_columns']
# SAMPLING_RATE = config['sampling_rate']

def data_load(data_file_path):

    """
    Load data from a CSV file and normalize the time column.
    """
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")

    # Load the data
    data = pd.read_csv(data_file_path)
    data = data[COLUMN_NAMES]
    return data



# 기본 경로 설정
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH /  'simulation_TOTAL'  # 시간이 일정한 데이터 파일: SYNC
# SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH /  'simulation_TOTAL'  # 시간이 일정한 데이터 파일: SYNC


cls_label = config['class_to_label']
label_cls = config['label_to_class']
short_to_long_label = config['Short_to_Long_Label']

label_data = pd.read_csv(SINGLE_SCENARIO_SYNLOG_DATA_ROOT / 'label.csv')
labeled_data = [[] for _ in range(len(cls_label))]

count = 0
for file, label in zip(label_data['file_name'], label_data['trajectory type']):

    # if label != 'right_lane_change':
    #     continue

    # if 'SSST' not in file:
    #     continue

    file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
    data = data_load(file_path)

    ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
    NO_LABEL = 0
    counting = 1

    LLC = detect_left_lane_change(data, duration_sec=0.8, threshold=0.25)
    RLC = detect_right_lane_change(data)
    ST = detect_straight(data)
    RT = detect_right_turn(data)
    LT = detect_left_turn(data)
    RA = detect_roundabout(data)
    UT = detect_u_turn(data)

    # UT -> LT -> RT -> LLC -> RLC -> RA -> ST // priority sequence
    labels = ['UT','RA', 'LT', 'RT', 'LLC', 'RLC',  'ST']
    values = [UT, RA, LT, RT, LLC, RLC, ST]

    for i, value in enumerate(values) :
        if value:
            result_list = [0] * 9
            short_label = labels[i]
            label_cls_index = label_cls[short_to_long_label[short_label]] - 1
            result_list[label_cls_index] = 1
            break  # 첫 번째 1만 인정

    # No label case
    if not any([LLC, RLC, ST, RT, LT, UT, RA]):
        result_list = [0] * 9
        result_list[-2] = 1

    result_list[-1] = counting

    # result_list = [ST, RT, LT, UT, LLC, RLC, RA, NO_LABEL, counting]
    labeled_data[label_cls[label]-1].append(result_list)
    # print(f'label: {label_cls[label]}, result: {result_list}')

    # df_copy = data.loc[:, ~data.columns.isin(['Entity'])].copy()
    # df_rolling = df_copy.rolling(100).mean().bfill()
    # df_rolling['time (sec)'] = df_rolling.index * 0.02 # index * 0.02

    # if './output/plots/RLC/' not in str(file):
    #     os.makedirs('./output/plots/RLC/', exist_ok=True)
    # time_base_plot(df_rolling, save_path= f'./output/plots/RLC/{file}_time_base_plot.png')
    # draw_ay_plot(df_rolling, save_path= f'./output/plots/RLC/{file}_ay_plot.png')

    # vel_x = df_rolling['VelocityX(EntityCoord) (km/h)'].values
    # vel_y = df_rolling['VelocityY(EntityCoord) (km/h)'].values
    # vel_z = df_rolling['VelocityZ(EntityCoord) (km/h)'].values
    # total_vel = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    # print(f'Velocity: {total_vel.mean():.2f} km/h')


    count += 1
    if count % 100 == 0:
        print(f'Processed {count} files...')
#%%
# total_result = []
# for i, label in enumerate(labeled_data):
#     np_sliced = np.array(label)
#     np_sliced_change = [[0 if x is None else x for x in row] for row in np_sliced]
#     column_sum = np.sum(np_sliced_change, axis=0)
#     total_result.append(column_sum)

# print(['ST', 'RT', 'LT', 'UT', 'LLC', 'RLC', 'RA', "NO_LABEL", "TOTAL"])
# df_data = pd.DataFrame(total_result)
# print(df_data)
#%%
total_result = []
for i, label in enumerate(labeled_data):
    np_sliced = np.array(label)
    np_sliced_change = [[0 if x is None else x for x in row] for row in np_sliced]
    column_sum = np.sum(np_sliced_change, axis=0)
    total_result.append(column_sum)

df_total_result = pd.DataFrame(total_result, columns=['ST', 'RT', 'LT', 'UT', 'LLC', 'RLC', 'RA', "NO_LABEL", "TOTAL"])
df_total_result.index = cls_label
print(df_total_result)

# %%
# UT -> LT -> RT -> LLC -> RLC -> RA -> ST