from rule_utils_masking.common_util import rolling

import numpy as np
import pandas as pd

def detect_roundabout_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    rolling_seconds: float = 2.0,
    threshold_neg: float = -0.3,
    threshold_pos: float = +0.3,
    duration_sec: float = 1.3,
    max_duration_sec: float = 8.0,
) -> np.ndarray:
    """
    라운드어바웃 패턴( pos_run -> neg_run -> pos_run )이 관찰되는
    전체 구간을 True로 마스킹하여 반환.

    - pos_run: ay >  threshold_pos 가 duration_sec 이상, max_duration_sec 이하 지속
    - neg_run: ay <  threshold_neg 가 duration_sec 이상, max_duration_sec 이하 지속
    - 순서: pos_run(이전) 시작 < neg_run 시작 < pos_run(이후) 시작

    반환:
      mask: np.ndarray(bool), len == len(df)
    """
    # 1) 스무딩(프로젝트의 rolling 사용 가정)
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].to_numpy()
    n = len(ay)
    if n == 0:
        return np.zeros(0, dtype=bool)

    min_frames = int(np.ceil(duration_sec * sampling_hz))
    max_frames = int(np.floor(max_duration_sec * sampling_hz)) if max_duration_sec is not None else None

    # True run 추출 유틸: 길이가 min_frames 이상이고(그리고 max_frames 이하면)인 run만 반영
    def _true_runs_bounded(cond: np.ndarray, min_f: int, max_f: int | None):
        runs = []
        i = 0
        L = len(cond)
        while i < L:
            if not cond[i]:
                i += 1
                continue
            j = i
            while j < L and cond[j]:
                j += 1
            length = j - i
            if length >= min_f and (max_f is None or length <= max_f):
                runs.append((i, j))  # [start, end) end는 배타
            i = j
        return runs

    # 2) 조건 배열 및 run 추출
    ay_pos = ay >  threshold_pos
    ay_neg = ay <  threshold_neg

    pos_runs = _true_runs_bounded(ay_pos, min_frames, max_frames)
    neg_runs = _true_runs_bounded(ay_neg, min_frames, max_frames)

    if not pos_runs or not neg_runs:
        return np.zeros(n, dtype=bool)

    # 3) pos(이전) - neg - pos(이후) 삼중 패턴 찾기
    #    첫 번째로 성립하는 삼중 패턴에 대해 [pos_before.start ~ pos_after.end) 마스킹
    mask = np.zeros(n, dtype=bool)

    # pos_runs를 시작 시각으로 정렬(이미 정렬되어 있음), neg_runs도 동일
    # 각 neg_run에 대해, 그 이전 pos_run과 이후 pos_run을 찾는다.
    pos_starts = [p[0] for p in pos_runs]

    for n_start, n_end in neg_runs:
        # 이전 pos_run: 시작이 neg 시작보다 작은 것 중 가장 늦은 것
        idx_before = np.searchsorted(pos_starts, n_start) - 1
        if idx_before < 0:
            continue
        pos_before = pos_runs[idx_before]

        # 이후 pos_run: 시작이 neg 시작보다 큰 것 중 가장 이른 것
        idx_after = np.searchsorted(pos_starts, n_start)
        if idx_after >= len(pos_runs):
            continue
        pos_after = pos_runs[idx_after]

        # 순서 확인: pos_before.start < neg.start < pos_after.start
        if pos_before[0] < n_start < pos_after[0]:
            s = pos_before[0]
            e = pos_after[1]
            mask[s:e] = True
            return mask  # 첫 패턴만 마스킹 (여러 패턴 원하면 누적 후 break 대신 continue)

    # 패턴이 없으면 전부 False
    return mask
