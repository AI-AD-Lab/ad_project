#%%
import pandas as pd
import numpy as np
from pathlib import Path

import os
import matplotlib.pyplot as plt
from _utils.data_processing_utils import normalize_time
from _utils.utils_plot import time_base_plot

from rule_utils.left_turn import *
from rule_utils.right_turn import *

from pathlib import Path

# from config import Config


random_seed = 42
np.random.seed(random_seed)
'''
used_columns: Index(['PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'RotationZ (deg)',
       'FrontWheelAngle (deg)', 'speed', 'acc']
'''

MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "rule_single_scenario_logs"       

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
    ['time (sec)', 'VelocityX(EntityCoord) (km/h)', 'VelocityY(EntityCoord) (km/h)']

    Returns:
    - df: ay, wz가 추가된 DataFrame
    """

    # 1. 속도 계산 (v = dx/dt)
    df['time (sec)'] = df.index * 0.016666
    dt = 1.0 / 60


    vx = df['VelocityX(EntityCoord) (km/h)'] * 1000 / 3600  # m/s
    vy = df['VelocityY(EntityCoord) (km/h)'] * 1000 / 3600  # m/s

    heading = np.arctan2(vy, vx)            # 차량 진행 방향
    heading_unwrapped = np.unwrap(heading)  # 불연속 보정
    wz = np.diff(heading_unwrapped, prepend=heading_unwrapped[0]) / dt      # yaw rate (rad/s)

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

def detect_llc_with_continuous_lateral_motion(df,
                                              lateral_threshold=2.0,
                                              consecutive_frames=60,
                                              steering_threshold=-0.05,
                                              hz=60):
    """
    곡선 도로에서도 적용 가능한 좌측 차선 변경(LLC) 감지

    Parameters:
    - df: DataFrame with ['time', 'x', 'y', 'front_wheel_angle']
    - lateral_threshold: 누적 좌측 이동 임계값 (m)
    - consecutive_frames: 몇 프레임 이상 연속으로 좌측 이동해야 하는지
    - steering_threshold: 조향각이 이 값보다 작으면 좌측 조향으로 판단

    Returns:
    - bool: True if LLC detected
    """
    dt = df['time (sec)'].diff().fillna(1 / hz)
    dx = df['PositionX (m)'].diff().fillna(0)
    dy = df['PositionY (m)'].diff().fillna(0)

    vx = dx / dt
    vy = dy / dt
    heading = np.arctan2(vy, vx)
    heading_unit = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    left_unit = np.stack([-np.sin(heading), np.cos(heading)], axis=1)

    # 매 프레임 이동벡터
    move = np.stack([dx, dy], axis=1)
    lateral_displacement = np.sum(move * left_unit, axis=1)

    # 누적 좌측 이동량
    total_left = lateral_displacement.sum()

    # 연속 좌측 이동 구간
    left_sequence = (lateral_displacement > 0).astype(int)
    count = 0
    max_seq = 0
    for val in left_sequence:
        if val:
            count += 1
            max_seq = max(max_seq, count)
        else:
            count = 0

    # 좌측 조향 존재 여부
    min_angle = df['FrontWheelAngle (deg)'].min()

    if (total_left >= lateral_threshold and
        max_seq >= consecutive_frames and
        min_angle < steering_threshold):
        
        print(f"✅ LLC 감지됨 | 누적 좌측이동: {total_left:.2f}m, 연속프레임: {max_seq}, 조향각: {min_angle:.2f}")
        return True

    print(f"❌ LLC 아님 | 누적 좌측이동: {total_left:.2f}m, 연속프레임: {max_seq}, 조향각: {min_angle:.2f}")
    return False

def estimate_curvature(df):
    vx = df['PositionX (m)'].diff().fillna(0) / df['time (sec)'].diff().fillna(1/60)
    vy = df['PositionY (m)'].diff().fillna(0) / df['time (sec)'].diff().fillna(1/60)
    heading = np.unwrap(np.arctan2(vy, vx))
    d_heading = heading.iloc[-1] - heading.iloc[0]
    dx_total = df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[0]
    dy_total = df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[0]
    arc_length = np.sqrt(dx_total**2 + dy_total**2)
    curvature = d_heading / arc_length
    return curvature, arc_length

def corrected_lateral_offset(lateral_offset, curvature, arc_length, gain=1.0):
    # 곡률로 인해 생겼을 법한 좌측 이동 보정
    expected_drift = curvature * arc_length * gain
    corrected = lateral_offset - expected_drift
    return corrected

def draw_wz_plot(df, save_path:None|str|Path = None):
    """
    Yaw Rate (wz)를 시간에 따라 시각화해주는 함수

    Parameters:
    - df: DataFrame containing wz와 time 값
    - time_col: 시간 컬럼 이름 (기본: 'time (sec)')
    - wz_col: wz 컬럼 이름 (기본: 'wz')
    - title: 그래프 제목
    """

    plt.figure(figsize=(10, 4))
    plt.plot(df['time (sec)'], df['wz'], label='Yaw Rate (wz)', color='tab:blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 시각적 가이드 라인 (논문 기준)
    # plt.axhline(0.02, color='green', linestyle=':', linewidth=1, label='LT threshold (+0.2 rad/s)')
    # plt.axhline(-0.02, color='red', linestyle=':', linewidth=1, label='RT threshold (−0.2 rad/s)')

    plt.xlabel('Time (sec)')
    plt.ylabel('Yaw Rate (rad/s)')
    plt.title('Yaw Rate (wz) over Time')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return

    plt.show()

def draw_ay_plot(df, save_path:None|str|Path = None):
    """
    Yaw Rate (wz)를 시간에 따라 시각화해주는 함수

    Parameters:
    - df: DataFrame containing wz와 time 값
    - time_col: 시간 컬럼 이름 (기본: 'time (sec)')
    - wz_col: wz 컬럼 이름 (기본: 'wz')
    - title: 그래프 제목
    """

    plt.figure(figsize=(10, 4))
    plt.plot(df['time (sec)'], df['AccelerationY(EntityCoord) (m/s2)'], label='Acc Y', color='tab:blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # 시각적 가이드 라인 (논문 기준)
    # plt.axhline(0.3, color='green', linestyle=':', linewidth=1, label='LT threshold (+0.3 rad/s)')
    # plt.axhline(-0.3, color='red', linestyle=':', linewidth=1, label='RT threshold (−0.3 rad/s)')

    plt.xlabel('Time (sec)')
    plt.ylabel('Yaw Rate (rad/s)')
    plt.title('Yaw Rate (ay) over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return 

    plt.show()


def detect_llc_by_ay_wz(df, ay_col='AccelerationY(EntityCoord) (m/s2)', wz_col='wz', sampling_hz=60,
                        ay_threshold=-0.5, wz_max_limit=0.1, duration_thresh=30):
    """
    ay/wz 기반 왼쪽 차선 변경(LLC) 감지 함수

    Parameters:
    - df: DataFrame containing 'ay' and 'wz'
    - sampling_hz: Hz 단위 (default: 60)
    - ay_threshold: ay가 이보다 작으면 좌측 변경 가능성
    - wz_max_limit: yaw rate가 이보다 크면 회전으로 간주 → 차선 변경 아님
    - duration_thresh: ay < ay_threshold 지속 프레임 수 기준 (default: 30 프레임 ≈ 0.5초)

    Returns:
    - True if LLC detected
    """

    ay = df[ay_col]
    wz = df[wz_col]

    # 조건 1: ay가 threshold보다 작게 내려간 프레임 수
    active_ay = (ay < ay_threshold).astype(int)
    max_seq = 0
    count = 0
    for val in active_ay:
        if val:
            count += 1
            max_seq = max(max_seq, count)
        else:
            count = 0

    # 조건 2: yaw rate가 너무 크지 않음 (회전이 아님)
    wz_max = wz.abs().max()

    if max_seq >= duration_thresh and wz_max < wz_max_limit:
        print(f"✅ LLC 감지됨 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
        return True

    print(f"❌ LLC 아님 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
    return False

output_save_dir = Path('./output2/accy_wz/')

for file in all_single_entity_file:
    file = Path(file)
    file_path = created_single_entity_dir / file
    base_name = file.stem
    save_path = output_save_dir / base_name

    ay_save = str(save_path) + '_ay.png'
    wz_save = str(save_path) + '_wz.png'


    data = pd.read_csv(file_path)
    data = normalize_time(data)
    data = data[180:]  

    # 데이터가 일정 속도 이상부터 시작하도록 필터링
    
    # detect_llc_with_continuous_lateral_motion(data)

    # time_base_plot(data)
    # detect_left_lane_change_vector(data)
    # print(f'left turn: {is_left_turn(data)}, right turn: {is_right_turn(data)}')


    dd = compute_ay_wz_from_xyz(data)
    dd.fillna(0)

    time_base_plot(dd)
    detect_left_lane_change_vector(dd)
    detect_llc_by_ay_wz(dd)
    # draw_ay_plot(dd)
    # draw_wz_plot(dd)

    
#     time_base_plot(data, save_path=f"./output2/wheel/{file.replace('.csv', '')}_plot.png")
    # label = rule_based_classifier(ex_data)

    # CSV 파일에 레이블 추가
    # label_df = pd.DataFrame({'Label': [label]})
    # label_df.to_csv(file_path, mode='a', header=False, index=False)
# %%
