#%%
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch

import os
import matplotlib.pyplot as plt
from _utils.data_processing_utils import normalize_time
from _utils.utils_plot import time_base_plot

from rule_utils.left_turn import *
from rule_utils.right_turn import *

# from config import Config


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
'''
used_columns: Index(['PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'RotationZ (deg)',
       'FrontWheelAngle (deg)', 'speed', 'acc']
'''

MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

# %%

import numpy as np
import pandas as pd

def detect_left_lane_change_vector(df,
                                   angle_threshold=-0.05,
                                   left_move_threshold=2.0,
                                   yaw_change_limit=-0.1):
    """
    차량의 로컬 좌표계 기준 왼쪽 차선 변경 감지

    Parameters:
    - df: trajectory DataFrame with [''PositionX (m)', 'position_y', 'RotationZ (deg)', 'front_wheel_angle']
    - angle_threshold: 조향각이 음수(왼쪽)로 꺾이는 임계값
    - left_move_threshold: 차량의 좌측으로 이동한 최소 거리 (m)
    - yaw_change_limit: 전체 yaw 변화가 회전 아닌 수준인지 확인 (deg)

    RotationZ (deg) : 차량의 회전각 (Yaw)

    Returns:
    - bool: 왼쪽 차선 변경이면 True
    """

    # 1. 진행 방향 벡터 계산 (초기 0~5프레임)
    alpha = 0.8
    dx_first = df['PositionX (m)'].iloc[180] - df['PositionX (m)'].iloc[0]
    dy_first = df['PositionY (m)'].iloc[180] - df['PositionY (m)'].iloc[0]
    heading_first = np.array([dx_first, dy_first])

    dx_last = df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[-180]
    dy_last = df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[-180]
    heading_last = np.array([dx_last, dy_last])

    # heading_norm = heading / (np.linalg.norm(heading) + 1e-8)

    dx = df['PositionX (m)'].diff().fillna(0)
    dy = df['PositionY (m)'].diff().fillna(0)
    mean_dx = dx.mean()
    mean_dy = dy.mean()
    heading_mean = np.array([mean_dx, mean_dy])
    

    heading_first_norm = heading_first / (np.linalg.norm(heading_first) + 1e-8)
    heading_last_norm = heading_last / (np.linalg.norm(heading_last) + 1e-8)
    heading_mean_norm = heading_mean / (np.linalg.norm(heading_mean) + 1e-8)

    heading = alpha * heading_first_norm + (1 - alpha) * heading_mean_norm

    # first와 last 차이
    heading_diff = np.arctan2(heading_last_norm[1], heading_last_norm[0]) - np.arctan2(heading_first_norm[1], heading_first_norm[0])
    heading_diff = np.degrees(heading_diff)
    print(f"Heading Diff: {heading_diff:.2f}°")

    # 2. 차량의 '왼쪽 방향' 벡터는 heading에 수직인 벡터
    # (x, y) → (-y, x) 로 90도 회전 (반시계방향)
    left_dir = np.array([-heading[1], heading[0]])

    # 3. 전체 이동 벡터
    move = np.array([
        df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[180],
        df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[180]
    ])

    # 4. 왼쪽 방향으로의 이동량 (벡터 투영)
    leftward_move = np.dot(move, left_dir)

    # 5. front_wheel_angle 조건
    min_angle = df['FrontWheelAngle (deg)'].min()
    angle_condition = min_angle < angle_threshold

    # 6. yaw 변화량 (회전이 아님을 보장)
    yaw_start = df['RotationZ (deg)'].iloc[0:180].mean()
    yaw_end = df['RotationZ (deg)'].iloc[-60:-1].mean()
    yaw_change = ((yaw_end - yaw_start + 180) % 360) - 180
    yaw_condition = yaw_change < yaw_change_limit

    if leftward_move > left_move_threshold and (angle_condition or yaw_condition):
        print(f"✅ 왼쪽 차선 변경 감지됨 | 좌측 이동량: {leftward_move:.2f}m | angle: {min_angle:.2f}°")
        return True

    print(f"❌ 차선 변경 아님 | 좌측 이동량: {leftward_move:.2f}m | angle: {min_angle:.2f}°, yaw_change: {yaw_change:.2f}°")
    return False



def compute_ay_wz_from_xyz(df: pd.DataFrame):
    """
    x, y, z 좌표 기반으로 lateral acceleration (ay)와 yaw rate (wz) 계산

    Parameters:
    - df: DataFrame with columns ['time', 'x', 'y', 'z']

    Returns:
    - df: ay, wz가 추가된 DataFrame
    """

    # 1. 속도 계산 (v = dx/dt)
    df['time (sec)'] = df.index * 0.016666
    dt = df['time (sec)'].diff().fillna(0)  # sec

    vx = df['VelocityX(EntityCoord) (km/h)'] * 1000 / 3600  # m/s
    vy = df['VelocityY(EntityCoord) (km/h)'] * 1000 / 3600  # m/s

    # 2. 가속도 계산 (a = dv/dt)
    ax = vx.diff().fillna(0) / dt
    ay_global = vy.diff().fillna(0) / dt

    # 3. heading (yaw) 계산
    heading = np.arctan2(vy, vx)  # rad
    heading_unwrapped = np.unwrap(heading)

    # # 4. angular velocity wz (yaw rate) = d(heading)/dt
    # d_heading = np.diff(heading_unwrapped, prepend=heading_unwrapped[0])
    # # d_heading = np.where(d_heading > np.pi, d_heading - 2 * np.pi, d_heading)
    # wz = d_heading / dt  # rad/s
    
    heading = np.arctan2(vy, vx)               # rad
    heading_unwrapped = np.unwrap(heading)     # unwrap BEFORE diff
    d_heading = np.diff(heading_unwrapped, prepend=heading_unwrapped[0])
    wz = d_heading / dt    

    # 5. 차량 진행 방향 벡터
    heading_unit = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    lateral_unit = np.stack([-np.sin(heading), np.cos(heading)], axis=1)

    # 6. global 가속도를 local 좌표계로 투영
    a_global = np.stack([ax, ay_global], axis=1)
    a_lateral = np.sum(a_global * lateral_unit, axis=1)  # ay

    df['ay'] = a_lateral
    df['wz'] = wz

    return df


def detect_llc_rule_based(df,
                          ay_threshold=1.0,
                          wz_min=0.2,
                          wz_max=0.4):
    """
    논문: Driving maneuver classification from time series data - rule based
    기반으로 Left Lane Change (LLC)를 감지

    Parameters:
    - df: DataFrame with columns ['ay', 'wz']

    Returns:
    - bool: True if LLC detected
    """
    ay = df['ay']
    wz = df['wz']

    # 조건 1: ay > 1.5 (lateral 가속도)
    ay_condition = (ay > ay_threshold).sum() > 0

    # 조건 2: wz in [0.2, 0.4] (angular velocity)
    wz_condition = ((wz >= wz_min) & (wz <= wz_max)).sum() > 0

    if ay_condition and wz_condition:
        print(f"✅ Left Lane Change 감지됨 | ay>{ay_threshold} 횟수: {ay_condition}, wz∈[0.2,0.4] 존재")
        return True
    else:
        print(f"❌ Left Lane Change 아님 | 조건 불충족 | ay>{ay_threshold} 횟수: {ay_condition}, \n wz value: {wz}")
        return False


def detect_right_lane_change_vector(df,
                                    angle_threshold=0.05,
                                    right_move_threshold=0.7,
                                    yaw_change_limit=10):
    """
    차량의 로컬 좌표계 기준 오른쪽 차선 변경 감지

    Parameters:
    - df: trajectory DataFrame with ['position_x', 'position_y', 'yaw', 'front_wheel_angle']
    - angle_threshold: 조향각이 양수(오른쪽)로 꺾이는 임계값
    - right_move_threshold: 차량의 오른쪽으로 이동한 최소 거리 (m)
    - yaw_change_limit: 회전량이 이 범위 이하면 회전 없는 직진 판단

    Returns:
    - bool: 오른쪽 차선 변경이면 True
    """

    # 1. 진행 방향 벡터 (초기)
    dx = df['PositionX (m)'].iloc[5] - df['PositionX (m)'].iloc[0]
    dy = df['PositionY (m)'].iloc[5] - df['PositionY (m)'].iloc[0]
    heading = np.array([dx, dy])
    heading_norm = heading / (np.linalg.norm(heading) + 1e-8)

    # 2. 오른쪽 방향 벡터: heading의 시계 방향 90도 회전 = (y, -x)
    right_dir = np.array([heading_norm[1], -heading_norm[0]])

    # 3. 전체 이동 벡터
    move = np.array([
        df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[0],
        df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[0]
    ])

    # 4. 오른쪽 방향 이동량 계산
    rightward_move = np.dot(move, right_dir)

    # 5. front wheel angle 조건 (양수 확인)
    max_angle = df['FrontWheelAngle (deg)'].max()
    angle_condition = max_angle > angle_threshold

    # 6. yaw 변화량 (회전이 아님을 보장)
    yaw_start = df['RotationZ (deg)'].iloc[0]
    yaw_end = df['RotationZ (deg)'].iloc[-1]
    yaw_change = ((yaw_end - yaw_start + 180) % 360) - 180
    yaw_condition = abs(yaw_change) < yaw_change_limit

    if rightward_move > right_move_threshold and angle_condition and yaw_condition:
        print(f"✅ 오른쪽 차선 변경 감지됨 | 우측 이동량: {rightward_move:.2f}m | angle: {max_angle:.2f}°, yaw_change: {yaw_change:.2f}°")
        return True

    print(f"❌ 차선 변경 아님 | 우측 이동량: {rightward_move:.2f}m | angle: {max_angle:.2f}°, yaw_change: {yaw_change:.2f}°")
    return False


created_single_entity_dir = SINGLE_SCENARIO_LOG_DATA / 'created_20250522174927_cutin'
all_single_entity_file = os.listdir(created_single_entity_dir)
all_single_entity_file = [file for file in all_single_entity_file if not file.endswith('label.csv')]


for file in all_single_entity_file:
    file_path = created_single_entity_dir / file
    data = pd.read_csv(file_path)
    data = normalize_time(data)
    data = data[180:]  

    # 데이터가 일정 속도 이상부터 시작하도록 필터링
    


    # time_base_plot(data)
    detect_left_lane_change_vector(data)
    # print(f'left turn: {is_left_turn(data)}, right turn: {is_right_turn(data)}')
    # dd = compute_ay_wz_from_xyz(data)

    # # time에 따른 ay, wz 시각화
    # plt.figure(figsize=(12, 6))
    # # plt.plot(dd['time (sec)'], dd['ay'], label='Lateral Acceleration (ay)')
    # plt.plot(dd['time (sec)'], dd['wz'], label='Yaw Rate (wz)')
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Value')
    # plt.title('Lateral Acceleration and Yaw Rate over Time')
    # plt.legend()
    # plt.grid()
    # # plt.savefig(f"./output2/wheel/{file.replace('.csv', '')}_ay_wz.png")
    # plt.show()


    # detect_llc_rule_based(dd)
    
#     time_base_plot(data, save_path=f"./output2/wheel/{file.replace('.csv', '')}_plot.png")
    # label = rule_based_classifier(ex_data)

    # CSV 파일에 레이블 추가
    # label_df = pd.DataFrame({'Label': [label]})
    # label_df.to_csv(file_path, mode='a', header=False, index=False)
# %%
