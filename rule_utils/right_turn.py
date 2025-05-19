import numpy as np
import pandas as pd

def is_right_turn(df: pd.DataFrame, angle_threshold=15, yaw_delta_threshold=20):
    """
    차량의 trajectory DataFrame을 받아 우회전 여부를 판단합니다.

    Parameters:
    - df: pd.DataFrame with columns ['time', 'position_x', 'position_y', 'yaw']
    - angle_threshold: 초기 이동 방향과 회전 궤적 사이 각도 차이 기준 (degrees)
    - yaw_delta_threshold: yaw의 누적 증가량 기준 (degrees)

    Returns:
    - bool: True if right turn detected
    """

    # 1. 초기 이동 방향 계산
    dx = df['PositionX (m)'].iloc[5] - df['PositionX (m)'].iloc[0]
    dy = df['PositionY (m)'].iloc[5] - df['PositionY (m)'].iloc[0]
    heading_angle = np.degrees(np.arctan2(dy, dx))  # 초기 주행 방향 (deg)

    # 2. 전체 이동 벡터
    dx_total = df['PositionX (m)'].iloc[-1] - df['PositionX (m)'].iloc[0]
    dy_total = df['PositionY (m)'].iloc[-1] - df['PositionY (m)'].iloc[0]
    final_angle = np.degrees(np.arctan2(dy_total, dx_total))  # 전체 주행 각도 (deg)

    # 3. 궤적 회전 각도 변화량 계산
    angle_diff = (final_angle - heading_angle + 360) % 360
    if angle_diff > 180:
        angle_diff -= 360  # [-180, +180] 범위로 정규화

    # 4. yaw 변화량 확인
    yaw_change = df['RotationZ (deg)'].iloc[-1] - df['RotationZ (deg)'].iloc[0]
    yaw_change = (yaw_change + 360) % 360
    if yaw_change > 180:
        yaw_change -= 360

    print(f"Angle Diff: {angle_diff}, Yaw Change: {yaw_change}")

    # 5. 판단 조건: 오른쪽으로 일정 각도 이상 회전 && yaw도 증가
    if abs(angle_diff) > 10 and angle_diff < 0 and yaw_change < yaw_delta_threshold:
        return True
    return False