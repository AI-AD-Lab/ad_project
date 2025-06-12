import pandas as pd
import numpy as np

def detect_roundabout(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_window = 100,
    threshold_neg=-0.3,
    threshold_pos=+0.3,
    duration_sec=1.3
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

    def find_starting_idxs(condition_array, right=False):
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

    # 조건 배열
    ay_neg = ay < threshold_neg # 임계값보다 낮은 경우 -> 왼쪽 가속도
    ay_pos = ay > threshold_pos # 임계값보다 높은 경우 -> 오른쪽 가속도

    # 최초 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)


    if neg_start and pos_start: # 왼쪽 회전 및 오른쪽 회전이 존재함
        if len(pos_start) >= 2: # 오른쪽 회전이 2번 탐지
            if min(pos_start) < min(neg_start) and max(pos_start) > min(neg_start):
                return True  # roundabout
            
    return False