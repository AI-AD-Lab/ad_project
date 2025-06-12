import numpy as np
import pandas as pd
# yaw의 증가는  좌회전, 감소는 오른쪽 회전으로 가정
# 동쪽기준 0도, 북쪽기준 90도, 서쪽기준 180 or -180도, 남쪽기준 270 or -90도
# 차량의 궤적을 기반으로 회전 방향을 판단하는 함수들

def is_left_turn(df: pd.DataFrame,
                 yaw_delta_threshold=70,
                 angle_diff_threshold=15,
                 stop_threshold=0.2,
                 stop_duration=0.05):
    """
    좌회전 판단 함수

    Parameters:
    - df: DataFrame with ['time', 'position_x', 'position_y', 'yaw', 'front_wheel_angle']
    - yaw_delta_threshold: yaw 변화량 (deg) 기준. 음수값 (좌회전은 yaw 감소)
    - angle_diff_threshold: 전체 궤적 회전 방향 각도. 음수면 좌회전
    - stop_threshold: 정지 상태로 판단할 속도 임계값 (m/s)
    - stop_duration: 정지로 간주할 최소 시간 (초)

    Returns:
    - bool: 좌회전이면 True
    """
    
    # 1. 속도 계산
    dx = df['PositionX (m)'].diff()
    dy = df['PositionY (m)'].diff()
    dt = df['time (sec)'].diff().replace(0, 1e-6)
    speed = np.sqrt(dx**2 + dy**2) / dt

    # 2. 정지 후 출발 여부 판단
    stopped = (speed < stop_threshold)
    stop_time = (df['time (sec)'][stopped].max() - df['time (sec)'][stopped].min()) if stopped.any() else 0
    has_stopped = stop_time >= stop_duration

    # 3. 전체 궤적 이동 방향 각도 차
    start_idx = df[df['time (sec)'] > 0.05].index[0]
    vec_start = np.array([
        df['PositionX (m)'].iloc[start_idx] - df['PositionX (m)'].iloc[0],
        df['PositionY (m)'].iloc[start_idx] - df['PositionY (m)'].iloc[0]
    ])
    vec_end = np.array([
        df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[0],
        df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[0]
    ])
    angle_start = np.degrees(np.arctan2(vec_start[1], vec_start[0]))
    angle_end = np.degrees(np.arctan2(vec_end[1], vec_end[0]))
    angle_diff = ((angle_end - angle_start + 180) % 360) - 180  # [-180, +180]

    # 4. yaw 변화량 (좌회전은 양수, 360 wraparound 고려)
    yaw_start = df['RotationZ (deg)'].iloc[0]
    yaw_end = df['RotationZ (deg)'].iloc[-1]
    yaw_change = ((yaw_end - yaw_start + 180) % 360) - 180

    print(f"Angle Diff: {angle_diff}, Yaw Change: {yaw_change}, Stopped: {has_stopped}")
    
    # 5. 좌회전 판단
    # 주 조건: 궤적 회전 각도 > 10 도 && yaw 변화량 > 80
    if angle_diff > angle_diff_threshold and yaw_change > yaw_delta_threshold:
        return True
 
    return False

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