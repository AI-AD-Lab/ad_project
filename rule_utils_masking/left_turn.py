# yaw의 증가는  좌회전, 감소는 오른쪽 회전으로 가정
# 동쪽기준 0도, 북쪽기준 90도, 서쪽기준 180 or -180도, 남쪽기준 270 or -90도
# 차량의 궤적을 기반으로 회전 방향을 판단하는 함수들

from rule_utils_masking.common_util import rolling
import numpy as np
import pandas as pd

from typing import Optional

def detect_left_turn_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    rolling_seconds: float = 2.0,
    left_threshold: float = -1.0,
    duration_sec: float = 3.0,
    max_duration_sec: Optional[float] = None,  # ✅ 상한(초) 추가. None이면 상한 미적용
) -> np.ndarray:
    """
    왼쪽 회전(LT) 구간 마스크 반환:
      - 조건: ay < left_threshold
      - 연속 길이가 [duration_sec, max_duration_sec] (max_duration_sec가 None이면 최소만)인 구간만 True
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
    cond = (ay < left_threshold)
    min_frames = int(np.ceil(duration_sec * sampling_hz))
    if max_duration_sec is None:
        max_frames = None
    else:
        max_frames = int(np.floor(max_duration_sec * sampling_hz))
        if max_frames < min_frames:  # 안전장치
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

        # True run: [i, j)
        j = i
        while j < n and cond[j]:
            j += 1
        run_len = j - i

        if max_frames is None:
            if run_len >= min_frames:
                out[i:j] = True
        else:
            if min_frames <= run_len <= max_frames:
                out[i:j] = True

        i = j

    return out

# def detect_left_turn_mask(
#     df: pd.DataFrame,
#     ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz: int = 50,
#     rolling_seconds: float = 2.0,
#     left_threshold: float = -1.0,
#     duration_sec: float = 3.0,
# ) -> np.ndarray:
#     """
#     왼쪽 회전(LT) 구간 마스크 반환:
#       - 조건: ay < left_threshold
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
#     cond = (ay < left_threshold)
#     min_frames = int(np.ceil(duration_sec * sampling_hz))
#     max_frames = int(np.ceil(8 * sampling_hz))

#     # 3) 연속 True run이 min_frames 이상인 구간만 유지
#     return _keep_runs_ge(cond, min_frames, max_frames)


# def _keep_runs_ge(cond: np.ndarray, min_frames: int, max_frames) -> np.ndarray:
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
#         if (j - i) >= min_frames and (j-i)<=max_frames:
#             out[i:j] = True
#         i = j
#     return out