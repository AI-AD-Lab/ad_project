from rule_utils.common_util import rolling

def detect_roundabout(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    threshold_neg=-0.3,
    threshold_pos=+0.3,
    duration_sec=1.3,
    max_duration_sec=8
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
    df_rolling =  rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)

    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz)
    max_frames = int(max_duration_sec * sampling_hz)

    # def find_starting_idxs(condition_array):
    #     # condition array is consist of True or False
    #     starting_points = []
    #     count = 0
    #     for i, cond in enumerate(condition_array):
    #         if cond:
    #             count += 1
    #             if count >= min_frames:
    #                 idx = i - count + 1
    #                 if idx not in starting_points:
    #                     starting_points.append(idx)
    #         else:
    #             count = 0

    #     return starting_points if starting_points else None

    def find_starting_idxs(condition_array):  # NOT SO GOOD PERFORMANCE
        starting_points = []
        count = 0
        current_start = None

        for i, cond in enumerate(condition_array):
            if cond:
                count += 1
                if count == min_frames:
                    current_start = i - count + 1
                    starting_points.append(current_start)
                elif count > max_frames:
                    # max_frames 초과 → 마지막에 넣은 시작점 제거
                    if current_start in starting_points:
                        starting_points.remove(current_start)
                    current_start = None
            else:
                count = 0
                current_start = None

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
                return 1  # roundabout

    return 0