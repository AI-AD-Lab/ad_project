from rule_utils_masking.common_util import rolling

import numpy as np
import pandas as pd
from typing import Optional

def detect_right_turn_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    rolling_seconds: float = 2.0,
    right_threshold: float = 1.0,
    duration_sec: float = 3.0,
    max_duration_sec: Optional[float] = None,  # ✅ 상한(초) 추가. None이면 상한 미적용
) -> np.ndarray:
    """
    오른쪽 회전(RT) 구간 마스크 반환:
      - 조건: ay > right_threshold
      - 위 조건이 duration_sec 이상 (그리고 max_duration_sec 이하일 때만) 연속 유지되는 모든 구간을 True로 표시
    반환:
      - mask: np.ndarray(bool), 길이 == len(df)
    """
    # 1) 스무딩 (프로젝트의 rolling 함수 사용 가정)
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].to_numpy()
    n = len(ay)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 2) 조건 및 프레임 변환
    cond = (ay > right_threshold)
    min_frames = int(np.ceil(duration_sec * sampling_hz))
    max_frames = None
    if max_duration_sec is not None:
        max_frames = int(np.floor(max_duration_sec * sampling_hz))
        # 안전장치: 상한이 하한보다 작게 들어오면 하한에 맞춰줌
        if max_frames < min_frames:
            max_frames = min_frames

    # 3) 연속 True run 길이가 [min_frames, max_frames] 범위만 유지
    return _keep_runs_between(cond, min_frames, max_frames)


def _keep_runs_between(cond: np.ndarray, min_frames: int, max_frames: Optional[int]) -> np.ndarray:
    """
    연속 True run의 길이가:
      - max_frames가 None이면: length >= min_frames
      - max_frames가 주어지면: min_frames <= length <= max_frames
    인 구간만 True로 유지.
    """
    if cond.size == 0:
        return cond
    out = np.zeros_like(cond, dtype=bool)
    i, n = 0, len(cond)

    while i < n:
        if not cond[i]:
            i += 1
            continue

        # True run 찾기: [i, j)
        j = i
        while j < n and cond[j]:
            j += 1
        run_len = j - i

        if max_frames is None:
            # 상한 없음: 최소 길이 이상이면 유지
            if run_len >= min_frames:
                out[i:j] = True
        else:
            # 상한 있음: [min, max] 범위만 유지
            if min_frames <= run_len <= max_frames:
                out[i:j] = True

        i = j

    return out

# def detect_right_turn_mask(
#     df: pd.DataFrame,
#     ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz: int = 50,
#     rolling_seconds: float = 2.0,
#     right_threshold: float = 1.0,
#     duration_sec: float = 3.0,
# ) -> np.ndarray:
#     """
#     오른쪽 회전(RT) 구간 마스크 반환:
#       - 조건: ay > right_threshold
#       - 위 조건이 duration_sec 이상 연속 유지되는 모든 구간을 True로 표시
#     반환:
#       - mask: np.ndarray(bool), 길이 == len(df)
#     """
#     # 1) 스무딩 (프로젝트의 rolling 함수 사용 가정)
#     df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
#     ay = df_rolling[ay_col].to_numpy()
#     n = len(ay)
#     if n == 0:
#         return np.zeros(0, dtype=bool)

#     # 2) 조건 및 최소 프레임
#     cond = (ay > right_threshold)
#     min_frames = int(np.ceil(duration_sec * sampling_hz))

#     # 3) 연속 True run이 min_frames 이상인 구간만 유지
#     return _keep_runs_ge(cond, min_frames)


# def _keep_runs_ge(cond: np.ndarray, min_frames: int) -> np.ndarray:
#     """연속 True run이 min_frames 이상인 구간만 True로 유지"""
#     if cond.size == 0:
#         return cond
#     out = np.zeros_like(cond, dtype=bool)
#     i, n = 0, len(cond)
#     while i < n:
#         if not cond[i]:
#             i += 1
#             continue
#         j = i
#         while j < n and cond[j]:
#             j += 1
#         # run = [i, j)
#         if (j - i) >= min_frames:
#             out[i:j] = True
#         i = j
#     return out
