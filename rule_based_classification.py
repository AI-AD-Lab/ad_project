#%%
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch

from dataset_single_entity import collate_fn_variable_length, LogDataset
from _utils.utils_path import sep_log_file, load_data
#%%
def extract_features(df):
    coords = df[['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']].values
    time = df['time (sec)'].values

    # 속도 및 가속도 계산
    dt = np.diff(time) + 1e-6  # 안정성 확보
    velocities = np.diff(coords, axis=0) / dt[:, None]  # (N-1)x3
    accels = np.diff(velocities, axis=0) / dt[:-1, None]  # (N-2)x3

    # 통계 기반 특징
    features = {
        'mean_accel_x': np.mean(accels[:, 0]),
        'mean_accel_y': np.mean(accels[:, 1]),
        'mean_accel_z': np.mean(accels[:, 2]),
        'std_accel_y': np.std(accels[:, 1]),
        'max_accel_y': np.max(accels[:, 1]),
        'min_accel_y': np.min(accels[:, 1])
    }
    return features

def rule_based_classifier(features):
    # 예시 룰: 가속 방향과 크기 기반 판단
    if features['max_accel_y'] > 1.5 and features['mean_accel_y'] > 0.5:
        return 'left_turn'
    elif features['min_accel_y'] < -1.5 and features['mean_accel_y'] < -0.5:
        return 'right_turn'
    elif features['mean_accel_x'] > 1.0:
        return 'acceleration'
    elif features['mean_accel_x'] < -1.0:
        return 'braking'
    else:
        return 'normal'
#%%

MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "test_scenario_logs"                  # From src
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

dataset = LogDataset(
    log_folder_path=SINGLE_SCENARIO_LOG_DATA, 
    simulation_folder='created_20250428150508', 
)

loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_variable_length)

# %%
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
'''
used_columns: Index(['PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'RotationZ (deg)',
       'FrontWheelAngle (deg)', 'speed', 'acc']
'''
path  = r"C:\Users\tndah\Desktop\STAGE_WS\single_scenario_logs\created_20250428150508\20250421_143857_R_KR_PR_Sangam_NoBuildings_tcar_t2_8_Ego.csv"
data = pd.read_csv(path)

# time 중복 제거
data = data.drop_duplicates(subset=['time (sec)'], keep='first')
data = data.reset_index(drop=True)
# unnamed columns 제거
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# %%
ex_data = extract_features(data)
print(ex_data)
# %%
# dataframe의 통계
# mean, std, max, min
data.describe()
#%%
# 시간에 따른 FrontWheelAngle 시각화
import matplotlib.pyplot as plt
from _utils.utils_plot import PLOTING, dataframe_2d_plot

def plot_front_wheel_angle(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['time (sec)'], data['FrontWheelAngle (deg)'], label='Front Wheel Angle', color='blue')
    plt.title('Front Wheel Angle Over Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Front Wheel Angle (deg)')
    plt.grid()
    plt.legend()
    plt.show()


# 한 그림에서 2*2 형태 그림
# 첫번째 그림은 wheel angle에 대한 그림
# 두번째 그림은 2D plot으로 x, y 좌표에 대한 그림
# 세번째 그림은 시간에 따른 x 좌표
# 네번째 그림은 시간에 따른 y 좌표

def plot_all(data, save_path:str|None=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 첫번째 그림: Front Wheel Angle
    axs[0, 0].plot(data['time (sec)'], data['FrontWheelAngle (deg)'], label='Front Wheel Angle', color='blue')
    axs[0, 0].set_title('Front Wheel Angle Over Time')
    axs[0, 0].set_xlabel('Time (sec)')
    axs[0, 0].set_ylabel('Front Wheel Angle (deg)')
    axs[0, 0].grid()
    axs[0, 0].legend()

    # 두번째 그림: 2D plot of x and y coordinates
    axs[0, 1].plot(data['PositionX (m)'], data['PositionY (m)'], label='Trajectory', color='green')
    axs[0, 1].set_title('2D Trajectory Plot')
    axs[0, 1].set_xlabel('PositionX (m)')
    axs[0, 1].set_ylabel('PositionY (m)')
    axs[0, 1].grid()
    axs[0, 1].legend()

    # 세번째 그림: Time vs PositionX
    axs[1, 0].plot(data['time (sec)'], data['PositionX (m)'], label='PositionX', color='red')
    axs[1, 0].set_title('Time vs PositionX')
    axs[1, 0].set_xlabel('Time (sec)')
    axs[1, 0].set_ylabel('PositionX (m)')
    axs[1, 0].grid()
    axs[1, 0].legend()

    # 네번째 그림: Time vs PositionY
    axs[1, 1].plot(data['time (sec)'], data['PositionY (m)'], label='PositionY', color='purple')
    axs[1, 1].set_title('Time vs PositionY')
    axs[1, 1].set_xlabel('Time (sec)')
    axs[1, 1].set_ylabel('PositionY (m)')
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.tight_layout()
    if save_path:
        if not save_path.endswith('.png'):
            save_path += '.png'
        plt.savefig(save_path)
    else:
        plt.show()

plot_all(data)
# %%
import os
all_single_entity_file = os.listdir(SINGLE_SCENARIO_LOG_DATA / 'created_20250428150508')
all_single_entity_file = [file for file in all_single_entity_file if not file.endswith('label.csv')]
# %%
for file in all_single_entity_file:
    file_path = SINGLE_SCENARIO_LOG_DATA/ 'created_20250428150508' / file
    data = pd.read_csv(file_path)
    data = data.drop_duplicates(subset=['time (sec)'], keep='first')
    data = data.reset_index(drop=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    ex_data = extract_features(data)

    plot_all(data, save_path=f"./output2/wheel/{file.replace('.csv', '')}_plot.png")
    # label = rule_based_classifier(ex_data)

    # CSV 파일에 레이블 추가
    # label_df = pd.DataFrame({'Label': [label]})
    # label_df.to_csv(file_path, mode='a', header=False, index=False)
# %%
