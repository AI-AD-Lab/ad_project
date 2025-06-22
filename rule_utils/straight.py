import pandas as pd
import numpy as np

def detect_straight(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_window = 100,
    abs_normal_threshold=0.05,
    abs_threshold=0.3, # 0.3? 0.6?
    duration_sec=6
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

    # 1단계 검증 -> ay의 변화가 거의 없는 경우 (직선길)
    ay_straight = abs(ay) < abs_normal_threshold
    if all(ay_straight):
        return 1

    # 2단계 검증 -> 일정 임계값 구역에서 일정 시간동안 유지되는 경우 (완만한 곡선)
    ay_first = ay[0]
    ay_diff = ay - ay_first
    ay_diff_straight = abs(ay_diff) < abs_threshold

    if all(ay_diff_straight):
        return 1

    # 3단계 검증 -> S자 형태 곡선 주행, 한쪽이라도 가속도가 지속적으로 유지되는 경우
    ay_neg = (ay < -abs_threshold)  # 음의 가속도
    ay_pos = (ay > abs_threshold)    # 양의 가속도

    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if neg_start and pos_start:
        return 1

    return 0