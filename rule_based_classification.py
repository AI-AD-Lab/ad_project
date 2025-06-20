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


'''
used_columns: Index(['PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'RotationZ (deg)',
       'FrontWheelAngle (deg)', 'speed', 'acc']
'''

# 기본 경로 설정
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH /  'TOTAL_SCNARIO'  # 시간이 일정한 데이터 파일: SYNC

# 세부 경로 설정

straight_dir_name_list = ['simulation_RA3_1', 'simulation_RA12_1', 'simulation_RA9_1' ]




driving_trajectory_dir_list = [
     SINGLE_SCENARIO_SYNLOG_DATA_ROOT / straight_dir_name_list[0],
                                SINGLE_SCENARIO_SYNLOG_DATA_ROOT / straight_dir_name_list[1],
                                SINGLE_SCENARIO_SYNLOG_DATA_ROOT / straight_dir_name_list[2]
                                ]

test_trajectory_dir_path = driving_trajectory_dir_list[0]
all_single_entity_file = os.listdir(test_trajectory_dir_path)
all_single_entity_file = [file for file in all_single_entity_file if file.endswith('_statelog.csv')]
print(all_single_entity_file[0])


#%%

import pandas as pd

    # 5. 좌회전 판단
    # 주 조건: 궤적 회전 각도 > 10 도 && yaw 변화량 > 80

def detect_left_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_window = 100,
    left_threshold=-1.0,
    duration_sec=3
):
    """
    ay만을 기반으로 좌우 차선 변경 판단 (지속적 ay 변화 기반)

    Parameters:
    - df: DataFrame containing ay
    - ay_col: lateral acceleration 컬럼명
    - sampling_hz: 데이터 주파수 (Hz)
    - threshold_neg: 음의 임계값 (RLC 후보)
    - threshold_pos: 양의 임계값 (LLC 후보)
    - duration_sec: 최소 지속 시간 (초 단위)
    """

    df_copy = df.loc[:, ~data.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(100).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz)

    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz)

    def find_starting_idxs(condition_array):
        # condition array is consist of True or False
        starting_points = []
        count = 0
        for i, cond in enumerate(condition_array):
            if cond:
                count += 1
                if count >= min_frames:
                    idx = i - count + 1
                    if idx not in starting_points:
                        starting_points.append(idx)
            else:
                count = 0

        return starting_points if starting_points else None


    ay_neg = ay < left_threshold # 임계값보다 낮은 경우 -> 왼쪽 가속도

    # 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)

    if neg_start:
        return True

    return False

def draw_rotz_plot(df, save_path:None|str|Path = None):
    """
    Yaw Rate (wz)를 시간에 따라 시각화해주는 함수

    Parameters:
    - df: DataFrame containing wz와 time 값
    - time_col: 시간 컬럼 이름 (기본: 'time (sec)')
    - wz_col: wz 컬럼 이름 (기본: 'wz')
    - title: 그래프 제목
    """

    plt.figure(figsize=(10, 4))
    plt.plot(df['time (sec)'], df['RotationZ (deg)'], label='Rotation Z', color='tab:blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 시각적 가이드 라인 (논문 기준)
    # plt.axhline(0.3, color='green', linestyle=':', linewidth=1, label='LT threshold (+0.3 rad/s)')
    # plt.axhline(-0.3, color='red', linestyle=':', linewidth=1, label='RT threshold (−0.3 rad/s)')

    # plt.xlabel('Time (sec)')
    # plt.ylabel('Rotation Z')
    # plt.title('Rotation Z over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return

    plt.show()

#%%
sampling_hz=50

total_V = []

for dir_path in driving_trajectory_dir_list:

    dir_path = Path(dir_path)
    scenario_all_log_files = os.listdir(dir_path)
    state_log_files = [statelog for statelog in scenario_all_log_files
                       if statelog.endswith('statelog.csv')]

    count = 0

    for file in state_log_files:

        file_path = dir_path / file


        data = pd.read_csv(file_path)

        # print(data.head())

        df_copy = data.loc[:, ~data.columns.isin(['Entity'])].copy()
        df_rolling = df_copy.rolling(100).mean().bfill()
        df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz)

        # result = detect_left_turn(data)
        # if not result:
        #     print(file_path)
        df_rolling['RotationZ (deg)'] =  np.unwrap(df_rolling['RotationZ (deg)'])
        # time_base_plot(data)
        draw_rotz_plot(df_rolling)

        # draw_ay_plot(df_rolling)

        count+=1
        if count >=5:
            break

# np_v = np.array(total_V)
# print(np_v.max())
# print(np_v.min())
# print(np_v.mean())


# %%

# not_detected_ra3_path = [
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180716_R_KR_PG_KATRI_RA3_1_130_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180958_R_KR_PG_KATRI_RA3_1_122_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180938_R_KR_PG_KATRI_RA3_1_123_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180837_R_KR_PG_KATRI_RA3_1_126_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180917_R_KR_PG_KATRI_RA3_1_124_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180757_R_KR_PG_KATRI_RA3_1_128_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_181018_R_KR_PG_KATRI_RA3_1_121_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180817_R_KR_PG_KATRI_RA3_1_127_statelog.csv',
# '/workspace/Stage/sample_scenario_logs_250610/simulation_RA3_1/20250608_180856_R_KR_PG_KATRI_RA3_1_125_statelog.csv'

# ]

# sampling_hz=50
# for file in not_detected_ra3_path:
#     data = pd.read_csv(file_path)

#     print(detect_roundabout(data))

#     df_copy = data.loc[:, ~data.columns.isin(['Entity'])].copy()
#     df_rolling = df_copy.rolling(100).mean().bfill()
#     df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz)

#     time_base_plot(data)
#     draw_ay_plot(df_rolling)