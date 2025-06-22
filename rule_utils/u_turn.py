import numpy as np
import pandas as pd


def detect_u_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_window = 100,
    threshold=2.0,
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


    ay_neg = ay < -threshold # 임계값보다 낮은 경우 -> 왼쪽 가속도

    # 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)

    if neg_start:
        return 1

    return 0