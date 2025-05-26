import numpy as np
import pandas as pd

def is_right_turn(df: pd.DataFrame, 
                  yaw_delta_threshold=-70,
                  angle_diff_threshold=-15,
                  ):
    """
    차량의 trajectory DataFrame을 받아 우회전 여부를 판단합니다.

    Parameters:
    - df: pd.DataFrame with columns ['time', 'position_x', 'position_y', 'yaw']
    - angle_threshold: 초기 이동 방향과 회전 궤적 사이 각도 차이 기준 (degrees)
    - yaw_delta_threshold: yaw의 누적 증가량 기준 (degrees)

    Returns:
    - bool: True if right turn detected
    """

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

    # 4. yaw 변화량 확인
    yaw_change = df['RotationZ (deg)'].iloc[-1] - df['RotationZ (deg)'].iloc[0]
    yaw_change = (yaw_change + 360) % 360
    if yaw_change > 180:
        yaw_change -= 360

    # print(f"Angle Diff: {angle_diff}, Yaw Change: {yaw_change}")

    # 5. 판단 조건: 오른쪽으로 일정 각도 이상 회전 && yaw도 증가
    if angle_diff < angle_diff_threshold and yaw_change < yaw_delta_threshold:
        return True
    return False