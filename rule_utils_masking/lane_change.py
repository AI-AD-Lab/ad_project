from rule_utils_masking.common_util import rolling
import numpy as np
import pandas as pd

def detect_left_lane_change_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    threshold: float = 0.1,
    rolling_seconds: float = 2.0,
    duration_sec: float = 0.8,
):
    """
    왼쪽 차선 변경(LLC) 규칙을 만족하는 구간을 True로 표시하는 마스크를 반환합니다.
    규칙:
      - ay < -threshold 가 duration_sec 이상 지속된 구간(neg_run)이 먼저 발생하고,
      - 이어서 ay > +threshold 가 duration_sec 이상 지속된 구간(pos_run)이 발생하면,
      - [neg_run.start  ~  pos_run.end] 구간을 LLC로 간주하고 True로 마스킹합니다.
    반환:
      - mask: np.ndarray(dtype=bool), len == len(df)
    """
    # 1) 스무딩(프로젝트의 rolling 함수 사용)
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    n = len(ay)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 유틸: 연속 True run 구간들(start, end)을 찾고,
    #       길이가 min_frames 이상인 run들만 반환
    def _true_runs_ge(cond: np.ndarray, min_frames: int):
        runs = []
        i = 0
        while i < len(cond):
            if not cond[i]:
                i += 1
                continue
            j = i
            while j < len(cond) and cond[j]:
                j += 1
            # run: [i, j)  (j는 처음으로 False가 된 지점)
            if (j - i) >= min_frames:
                runs.append((i, j))  # end는 배타적
            i = j
        return runs  # [(start_idx, end_idx_exclusive), ...]

    def _build_mask_from_pair(neg_run, pos_run):
        mask = np.zeros(n, dtype=bool)
        # 이벤트 범위: neg_run.start ~ pos_run.end (배타 인덱스 pos_run[1])
        s = neg_run[0]
        e = pos_run[1]
        mask[s:e] = True
        return mask

    # 최초 파라미터
    min_frames = int(np.ceil(duration_sec * sampling_hz))

    # 시도 1: 원 파라미터
    ay_neg = (ay < -threshold)
    ay_pos = (ay >  threshold)
    neg_runs = _true_runs_ge(ay_neg, min_frames)
    pos_runs = _true_runs_ge(ay_pos, min_frames)

    def _first_pair_mask(neg_runs, pos_runs):
        if not neg_runs or not pos_runs:
            return None
        # 원 코드 로직을 존중: "가장 이른 neg_run.start"와 "가장 이른 pos_run.start" 비교
        first_neg = neg_runs[0]
        first_pos = pos_runs[0]
        if first_neg[0] < first_pos[0]:
            return _build_mask_from_pair(first_neg, first_pos)
        return None  # 순서가 아니면 LLC로 보지 않음

    mask = _first_pair_mask(neg_runs, pos_runs)

    # 시도 2: 완화( threshold /= 2, duration *= 1.2 )
    if mask is None:
        thr2 = threshold / 2.0
        min_frames2 = int(np.ceil(min_frames * 1.2))
        ay_neg2 = (ay < -thr2)
        ay_pos2 = (ay >  thr2)
        neg_runs2 = _true_runs_ge(ay_neg2, min_frames2)
        pos_runs2 = _true_runs_ge(ay_pos2, min_frames2)
        mask = _first_pair_mask(neg_runs2, pos_runs2)

    # 없으면 전부 False
    if mask is None:
        return np.zeros(n, dtype=bool)
    return mask

def detect_right_lane_change_mask(
    df: pd.DataFrame,
    ay_col: str = 'AccelerationY(EntityCoord) (m/s2)',
    sampling_hz: int = 50,
    threshold: float = 0.1,
    rolling_seconds: float = 2.0,
    duration_sec: float = 1.0,
) -> np.ndarray:
    """
    오른쪽 차선 변경(RLC) 규칙을 만족하는 구간을 True로 표시하는 마스크를 반환합니다.
    규칙:
      - ay > +threshold 가 duration_sec 이상 지속된 구간(pos_run)이 먼저 발생하고,
      - 이어서 ay < -threshold 가 duration_sec 이상 지속된 구간(neg_run)이 발생하면,
      - [pos_run.start  ~  neg_run.end] 구간을 RLC로 간주하고 True로 마스킹합니다.
    반환:
      - mask: np.ndarray(dtype=bool), len == len(df)
    """
    # 1) 스무딩 (프로젝트의 rolling 함수 사용)
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    n = len(ay)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 유틸: 연속 True run 중 길이가 min_frames 이상인 구간만 수집
    def _true_runs_ge(cond: np.ndarray, min_frames: int):
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
            if (j - i) >= min_frames:
                runs.append((i, j))  # [start, end) end는 배타
            i = j
        return runs

    def _build_mask_from_pair(pos_run, neg_run):
        mask = np.zeros(n, dtype=bool)
        s = pos_run[0]
        e = neg_run[1]
        mask[s:e] = True
        return mask

    # 최초 파라미터
    min_frames = int(np.ceil(duration_sec * sampling_hz))

    # 시도 1: 원 파라미터 (pos 먼저, 그 다음 neg)
    ay_pos = (ay >  threshold)
    ay_neg = (ay < -threshold)
    pos_runs = _true_runs_ge(ay_pos, min_frames)
    neg_runs = _true_runs_ge(ay_neg, min_frames)

    def _first_pair_mask(pos_runs, neg_runs):
        if not pos_runs or not neg_runs:
            return None
        first_pos = pos_runs[0]
        first_neg = neg_runs[0]
        # pos가 먼저 시작해야 RLC
        if first_pos[0] < first_neg[0]:
            return _build_mask_from_pair(first_pos, first_neg)
        return None

    mask = _first_pair_mask(pos_runs, neg_runs)

    # 시도 2: 완화 (threshold /= 2, duration *= 1.2)
    if mask is None:
        thr2 = threshold / 2.0
        min_frames2 = int(np.ceil(min_frames * 1.2))
        ay_pos2 = (ay >  thr2)
        ay_neg2 = (ay < -thr2)
        pos_runs2 = _true_runs_ge(ay_pos2, min_frames2)
        neg_runs2 = _true_runs_ge(ay_neg2, min_frames2)
        mask = _first_pair_mask(pos_runs2, neg_runs2)

    # 없으면 전부 False
    if mask is None:
        return np.zeros(n, dtype=bool)
    return mask