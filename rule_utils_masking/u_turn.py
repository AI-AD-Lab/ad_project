from rule_utils_masking.common_util import rolling
import numpy as np
import pandas as pd

def detect_u_turn_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    rolling_seconds: float = 2.0,
    threshold: float = 2.0,
    duration_sec: float = 3.0,
) -> np.ndarray:
    """
    U-turn 구간 마스크 반환:
      - 조건: ay < -threshold
      - 위 조건이 duration_sec 이상 연속 유지되는 모든 구간을 True로 표시
    반환:
      - mask: np.ndarray(bool), 길이 == len(df)
    """
    # 1) 스무딩 (프로젝트의 rolling 함수 사용 가정)
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].to_numpy()
    n = len(ay)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 2) 조건 및 최소 프레임
    cond = (ay < -threshold)
    min_frames = int(np.ceil(duration_sec * sampling_hz))

    # 3) 연속 True run이 min_frames 이상인 구간만 True로 유지
    return _keep_runs_ge(cond, min_frames)


def _keep_runs_ge(cond: np.ndarray, min_frames: int) -> np.ndarray:
    """연속 True run이 min_frames 이상인 구간만 True로 유지"""
    if cond.size == 0:
        return cond
    out = np.zeros_like(cond, dtype=bool)
    i, n = 0, len(cond)
    while i < n:
        if not cond[i]:
            i += 1
            continue
        j = i
        while j < n and cond[j]:
            j += 1
        # run = [i, j)
        if (j - i) >= min_frames:
            out[i:j] = True
        i = j
    return out
