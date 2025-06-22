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
        return 1
    return 0

def detect_right_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_window = 100,
    right_threshold=1.0,
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

    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
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


    ay_neg = ay > right_threshold # 임계값보다 큰 경우 -> 오른쪽 가속도

    # 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)

    if neg_start:
        return 1

    return 0