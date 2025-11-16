from rule_utils_masking.common_util import rolling

import numpy as np
import pandas as pd

def detect_straight_mask(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    abs_normal_threshold=0.1,
    abs_threshold=0.3,
    duration_sec=8,
):
    """
    직선(ST) 주행 구간 마스크 반환
    - ay 절댓값이 작고 안정적인 구간
    - 완만한 곡선 또는 S자 구간 포함 가능
    """
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    n = len(ay)
    min_frames = int(duration_sec * sampling_hz)

    mask = np.zeros(n, dtype=bool)

    # Helper: 연속 True 구간 찾기
    def find_continuous_regions(condition_array, min_len):
        regions = []
        count = 0
        start = None
        for i, cond in enumerate(condition_array):
            if cond:
                if count == 0:
                    start = i
                count += 1
            else:
                if count >= min_len:
                    regions.append((start, i))
                count = 0
        # 마지막 구간 처리
        if count >= min_len:
            regions.append((start, n))
        return regions

    # 1️⃣ ay 변화 거의 없는 구간 (절대값 작음)
    mask_level1 = np.abs(ay) <= abs_normal_threshold
    regions1 = find_continuous_regions(mask_level1, min_frames)
    for s, e in regions1:
        mask[s:e] = True

    # 2️⃣ 일정 임계값 내에서 완만한 변화 구간
    ay_first = ay[0]
    ay_diff = np.abs(ay - ay_first) <= abs_normal_threshold * 2
    regions2 = find_continuous_regions(ay_diff, min_frames)
    for s, e in regions2:
        mask[s:e] = True

    # 3️⃣ 한쪽 가속도가 일정하게 유지되는 구간 (완만한 S자 포함)
    ay_neg = (ay < -abs_threshold)
    ay_pos = (ay > abs_threshold)

    neg_regions = find_continuous_regions(ay_neg, min_frames)
    pos_regions = find_continuous_regions(ay_pos, min_frames)

    for s, e in neg_regions + pos_regions:
        mask[s:e] = True

    return mask


def detect_straight_mask_flexible(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    window_sec=2.0,
    abs_normal_threshold=0.06,
    abs_threshold=0.12,
    duration_sec=10.0,
    min_coverage=0.7,
    balance_eps=0.2,
    exclude_ra_mask=None
):
    """
    유연한 직선 주행(ST) 탐지
    완만한 곡선이나 S자 주행도 ST로 분류 가능

    - ay 진폭이 작고,
    - ay의 평균이 거의 0에 가까우며,
    - 양/음 비율이 비슷할 때 ST로 인식
    """
    ay = df[ay_col].to_numpy()
    n = len(ay)
    win = int(window_sec * sampling_hz)
    min_frames = int(duration_sec * sampling_hz)
    if n < win:
        return np.zeros(n, dtype=bool)

    # 이동평균 기반 smoothing
    ay_smooth = np.convolve(ay, np.ones(win)/win, mode='same')

    mask = np.zeros(n, dtype=bool)
    for i in range(0, n - min_frames):
        seg = ay_smooth[i:i + min_frames]
        mean_abs = np.mean(np.abs(seg))
        pos_ratio = np.sum(seg > 0) / len(seg)
        neg_ratio = np.sum(seg < 0) / len(seg)

        # 조건 1: 절대값 평균이 작고 (작은 횡가속)
        cond1 = mean_abs < abs_threshold

        # 조건 2: 양/음 비율이 거의 균형
        cond2 = abs(pos_ratio - neg_ratio) < balance_eps

        # 조건 3: 대부분이 안정적 (작은 가속도)
        cond3 = np.mean(np.abs(seg) < abs_normal_threshold) > min_coverage

        if cond1 and cond2 and cond3:
            mask[i:i + min_frames] = True

    # RA가 주어진 경우 제외
    if exclude_ra_mask is not None and exclude_ra_mask.shape == mask.shape:
        mask &= ~exclude_ra_mask

    return mask

