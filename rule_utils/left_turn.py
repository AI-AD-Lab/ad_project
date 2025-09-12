# yaw의 증가는  좌회전, 감소는 오른쪽 회전으로 가정
# 동쪽기준 0도, 북쪽기준 90도, 서쪽기준 180 or -180도, 남쪽기준 270 or -90도
# 차량의 궤적을 기반으로 회전 방향을 판단하는 함수들

from rule_utils.common_util import rolling

def detect_left_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    left_threshold=-1.0,
    duration_sec=3
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


    ay_neg = ay < left_threshold # 임계값보다 낮은 경우 -> 왼쪽 가속도

    # 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)

    if neg_start:
        return 1

    return 0