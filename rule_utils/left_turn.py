# yaw의 증가는  좌회전, 감소는 오른쪽 회전으로 가정
# 동쪽기준 0도, 북쪽기준 90도, 서쪽기준 180 or -180도, 남쪽기준 270 or -90도
# 차량의 궤적을 기반으로 회전 방향을 판단하는 함수들

from rule_utils.common_util import rolling

# def detect_left_turn(
#     df,
#     ay_col='AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz=50,
#     rolling_seconds=2,
#     left_threshold=-1.0,
#     duration_sec=3,
#     max_duration_sec = 8
# ):
#     """
#     ay만을 기반으로 좌우 차선 변경 판단 (지속적 ay 변화 기반)

#     Parameters:
#     - df: DataFrame containing ay
#     - ay_col: lateral acceleration 컬럼명
#     - sampling_hz: 데이터 주파수 (Hz)
#     - threshold_neg: 음의 임계값 (RLC 후보)
#     - threshold_pos: 양의 임계값 (LLC 후보)
#     - duration_sec: 최소 지속 시간 (초 단위)
#     """

#     df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
#     ay = df_rolling[ay_col].values
#     min_frames = int(duration_sec * sampling_hz)
#     max_frames = int(max_duration_sec * sampling_hz)

#     def find_starting_idxs(condition_array):
#         # condition array is consist of True or False
#         starting_points = []
#         count = 0
#         for i, cond in enumerate(condition_array):
#             if cond:
#                 count += 1
#                 if count >= min_frames and count <= max_frames:
#                     idx = i - count + 1
#                     if idx not in starting_points:
#                         starting_points.append(idx)
#             else:
#                 count = 0

#         return starting_points if starting_points else None


#     ay_neg = ay < left_threshold # 임계값보다 낮은 경우 -> 왼쪽 가속도

#     # 이벤트 인덱스 탐지
#     neg_start = find_starting_idxs(ay_neg)

#     if neg_start:
#         return 1

#     return 0


# def detect_left_turn(
#     df,
#     ay_col='AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz=50,
#     rolling_seconds=2,
#     left_threshold=-1.0,
#     duration_sec=3,
#     max_duration_sec=8
# ):
#     """
#     ay만을 기반으로 좌측 차선 변경 판단 (지속적 ay 변화 기반)
#     - 이벤트 지속 시간이 [duration_sec, max_duration_sec] 범위일 때만 검출
#     """
#     df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
#     ay = df_rolling[ay_col].values
#     min_frames = int(duration_sec * sampling_hz)
#     max_frames = int(max_duration_sec * sampling_hz)

#     ay_neg = ay < left_threshold  # 임계값보다 낮은 경우 -> 왼쪽 가속도(True)

#     # 연속 구간 기반으로 길이 평가
#     starts = []
#     in_run = False
#     run_start = 0
#     run_len = 0

#     for i, cond in enumerate(ay_neg):
#         if cond:
#             if not in_run:
#                 in_run = True
#                 run_start = i
#                 run_len = 1
#             else:
#                 run_len += 1
#         else:
#             if in_run:
#                 # 구간 종료 -> 최종 길이 확정
#                 if min_frames <= run_len <= max_frames:
#                     starts.append(run_start)
#                 in_run = False
#                 run_len = 0

#     # 시퀀스가 True로 끝난 경우 마지막 구간 처리
#     if in_run:
#         if min_frames <= run_len <= max_frames:
#             starts.append(run_start)

#     # 하나라도 있으면 이벤트 존재로 간주
#     return 1 if starts else 0

def detect_left_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    left_threshold=-1.0,
    duration_sec=3,
    max_duration_sec=8
):
    """
    ay만을 기반으로 좌측 차선 변경 판단 (지속적 ay 변화 기반)
    - 이벤트 지속 시간이 [duration_sec, max_duration_sec] 범위일 때만 검출
    """
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz)
    max_frames = int(max_duration_sec * sampling_hz)

    ay_neg = ay < left_threshold  # 임계값보다 낮은 경우 -> 왼쪽 가속도(True)

    starts = []
    in_run = False
    run_start = 0
    run_len = 0

    for i, cond in enumerate(ay_neg):
        if cond:
            if not in_run:
                in_run = True
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            if in_run:
                # 구간 종료 -> 최종 길이 확정
                if min_frames <= run_len <= max_frames:
                    starts.append(run_start)
                in_run = False
                run_len = 0

    # 시퀀스가 True로 끝난 경우 마지막 구간 처리
    if in_run:
        if min_frames <= run_len <= max_frames:
            starts.append(run_start)

    # ✅ 하나라도 있고, 모두 상한 이하인 경우만 이벤트 존재로 간주
    if not starts:
        return 0

    # 혹시라도 어떤 구간이 max_frames 초과였다면 0으로 반환
    # (위에서 이미 필터링했으므로, starts가 비어 있지 않으면 안전하지만
    #  2중 안전장치로 한 번 더 확인)
    for i, cond in enumerate(ay_neg):
        if cond:
            run_len += 1
            if run_len > max_frames:
                return 0

    return 1