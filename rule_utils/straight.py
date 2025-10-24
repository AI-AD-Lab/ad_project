from rule_utils.common_util import rolling

def detect_straight(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    abs_normal_threshold=0.1,
    abs_threshold=0.3, # 0.3? 0.6?
    duration_sec=8
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

    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
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
    ay_straight = abs(ay) <= abs_normal_threshold
    if all(ay_straight):
        return 1

    # 2단계 검증 -> 일정 임계값 구역에서 일정 시간동안 유지되는 경우 (완만한 곡선)
    ay_first = ay[0]
    ay_diff = ay - ay_first
    ay_diff_straight = abs(ay_diff) <= abs_normal_threshold*2

    if all(ay_diff_straight):
        return 1

    # 3단계 검증 -> S자 형태 곡선 주행, 한쪽이라도 가속도가 지속적으로 유지되는 경우
    ay_neg = (ay < -abs_threshold)  # 음의 가속도
    ay_pos = (ay > abs_threshold)    # 양의 가속도

    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if neg_start or pos_start:
        return 1

    return 0


def refactorying_detect_straight(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    abs_normal_threshold=0.1,
    abs_threshold=0.3,  # 0.3? 0.6?
    duration_sec=8
):
    """
    ay(횡가속도) 기반 직선/완만한 곡선 구간 탐지.
    결과: 직선(또는 완만한 곡선 포함으로 간주) -> 1, 아니면 0
    """

    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz)

    def find_starting_idxs(condition_array):
        """True가 min_frames 이상 연속된 구간의 시작 인덱스 목록을 반환.
        없으면 None 반환 (기존 동작과 동일한 진리값 평가를 위해)."""
        starts = []
        count = 0
        for i, cond in enumerate(condition_array):
            if cond:
                count += 1
                if count >= min_frames:
                    idx = i - count + 1
                    if idx not in starts:
                        starts.append(idx)
            else:
                count = 0
        return starts if starts else None

    # 1) ay의 절대값이 모두 매우 작으면 직선으로 판단
    if all(abs(ay) <= abs_normal_threshold):
        return 1

    # 2) 시작 시점 대비 ay 변화폭이 매우 작으면(완만한 곡선) 직선으로 판단
    ay_diff = ay - ay[0]
    if all(abs(ay_diff) <= abs_normal_threshold * 2):
        return 1

    # 3) S자 등에서 한쪽 방향의 가속도가 일정 시간 이상 지속되면 직선(완만 포함)으로 판단
    ay_neg = ay < -abs_threshold
    ay_pos = ay >  abs_threshold

    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if neg_start or pos_start:
        return 1

    return 0
