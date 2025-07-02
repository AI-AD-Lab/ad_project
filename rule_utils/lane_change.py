import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def detect_llc_by_ay_wz(df, ay_col='AccelerationY(EntityCoord) (m/s2)',
                        sampling_hz=50, ay_threshold=-0.5, wz_max_limit=0.8, duration_thresh=30):
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


    if max_seq >= duration_thresh: #and wz_max < wz_max_limit:
        print(f"✅ LLC 감지됨 | ay 지속 프레임: {max_seq},")
        return True

    print(f"❌ LLC 아님 | ay 지속 프레임: {max_seq}, ")
    return False

def detect_lane_change_by_ay_direction(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    threshold_neg=-0.3,
    threshold_pos=+0.3,
    duration_sec=1
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

    Returns:
    - 'LLC' | 'RLC' | 'None'
    """

    ay = df[ay_col].values
    min_frames = int(duration_sec * sampling_hz)

    def find_first_event(condition_array):
        count = 0
        serial = 0
        for i, cond in enumerate(condition_array):
            if cond:
                count += 1
                if count >= min_frames and count > serial:
                    serial = count
                    idx = i - count + 1
            else:
                count = 0

        if serial != 0:
            return serial
        return None

    # 조건 배열
    ay_neg = ay < - threshold_neg #
    ay_pos = ay > threshold_pos

    # 최초 이벤트 인덱스 탐지
    neg_start = find_first_event(ay_neg)
    pos_start = find_first_event(ay_pos)

    if (neg_start is None) or (pos_start is None):
        return 'None'

    if min(neg_start) < min(pos_start):
        return "RLC"
    elif min(pos_start) < min(neg_start):
        return "LLC"

def detect_left_lane_change(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    threshold=0.1,
    rolling_window = 100,
    duration_sec=0.8
):
    # data smoothing, reduce noise
    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02

    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz) # duration in frames

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

        return starting_points if starting_points else -1

    # 조건 배열
    ay_neg = (ay < -threshold) #& (ay > -1.0)
    ay_pos = (ay > threshold)

    # 최초 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if (neg_start == -1) or (pos_start == -1):
        threshold /= 2
        min_frames *= 1.2
        ay_neg = (ay < -threshold)
        ay_pos = (ay > threshold) #& (ay < 1.0)

        neg_start = find_starting_idxs(ay_neg)
        pos_start = find_starting_idxs(ay_pos)

        if (neg_start == -1) or (pos_start == -1):
            return 0

    if min(neg_start) < min(pos_start):
        return 1
    elif min(pos_start) < min(neg_start):
        return 0
    else:
        return 0

def detect_right_lane_change(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    threshold=0.1,
    rolling_window = 100,
    duration_sec=1
):
    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02

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

        return starting_points if starting_points else -1

    # 조건 배열
    ay_neg = (ay < -threshold)
    ay_pos = (ay > threshold) #& (ay < 1.0)

    # 최초 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if (neg_start == -1) or (pos_start == -1):
        threshold /= 2
        min_frames *= 1.2
        ay_neg = (ay < -threshold)
        ay_pos = (ay > threshold) #& (ay < 1.0)

        neg_start = find_starting_idxs(ay_neg)
        pos_start = find_starting_idxs(ay_pos)

        if (neg_start == -1) or (pos_start == -1):
            return 0

    if min(neg_start) < min(pos_start):
        return 0
    elif min(pos_start) < min(neg_start):
        return 1
    else:
        return 0