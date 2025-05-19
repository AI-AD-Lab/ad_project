import numpy as np
import pandas as pd
# yaw의 증가는  좌회전, 감소는 오른쪽 회전으로 가정
# 동쪽기준 0도, 북쪽기준 90도, 서쪽기준 180 or -180도, 남쪽기준 270 or -90도
# 차량의 궤적을 기반으로 회전 방향을 판단하는 함수들

def is_left_turn(df: pd.DataFrame,
                 yaw_delta_threshold=-20,
                 angle_diff_threshold=-10,
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

    # 4. yaw 변화량 (좌회전은 음수, 360 wraparound 고려)
    yaw_start = df['RotationZ (deg)'].iloc[0]
    yaw_end = df['RotationZ (deg)'].iloc[-1]
    yaw_change = ((yaw_end - yaw_start + 180) % 360) - 180

    print(f"Angle Diff: {angle_diff}, Yaw Change: {yaw_change}, Stopped: {has_stopped}")
    
    # 5. 좌회전 판단
    if angle_diff > angle_diff_threshold and yaw_change > yaw_delta_threshold:
        return True

    # 보조 조건: 정지 후 출발 + 방향 회전 감지
    if has_stopped and angle_diff < -5 and yaw_change < -10:
        return True

    return False