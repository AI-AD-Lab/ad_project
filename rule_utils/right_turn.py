from rule_utils.common_util import rolling

# def detect_right_turn(
#     df,
#     ay_col='AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz=50,
#     rolling_seconds=2,
#     right_threshold=1.0,
#     duration_sec=3,
#     max_ducration_sec=8
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
#     max_frames = int(max_ducration_sec * sampling_hz)

#     def find_starting_idxs(condition_array):
#         # condition array is consist of True or False
#         starting_points = []
#         count = 0
#         for i, cond in enumerate(condition_array):
#             if cond:
#                 count += 1
#                 if count >= min_frames:
#                     idx = i - count + 1
#                     if idx not in starting_points:
#                         starting_points.append(idx)

#             else:
#                 count = 0

#         return starting_points if starting_points else None


#     ay_neg = ay > right_threshold # 임계값보다 큰 경우 -> 오른쪽 가속도

#     # 이벤트 인덱스 탐지
#     neg_start = find_starting_idxs(ay_neg)

#     if neg_start:
#         return 1

#     return 0


# def detect_right_turn(
#     df,
#     ay_col='AccelerationY(EntityCoord) (m/s2)',
#     sampling_hz=50,
#     rolling_seconds=2,
#     right_threshold=1.0,
#     duration_sec=3,
#     max_duration_sec=8
# ):
#     """
#     ay만을 기반으로 우측 차선 변경 판단 (지속적 ay 변화 기반)
#     - 이벤트 지속 시간이 [duration_sec, max_duration_sec] 범위일 때만 검출
#     """
#     df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
#     ay = df_rolling[ay_col].values

#     min_frames = int(duration_sec * sampling_hz)
#     max_frames = int(max_duration_sec * sampling_hz)

#     ay_pos = ay > right_threshold  # 임계값보다 큰 경우 -> 오른쪽 가속도(True)

#     starts = []
#     in_run = False
#     run_start = 0
#     run_len = 0

#     for i, cond in enumerate(ay_pos):
#         if cond:
#             if not in_run:
#                 in_run = True
#                 run_start = i
#                 run_len = 1
#             else:
#                 run_len += 1
#         else:
#             if in_run:
#                 # 구간 종료 -> 최종 길이 확정 후 범위 판정
#                 if min_frames <= run_len <= max_frames:
#                     starts.append(run_start)
#                 in_run = False
#                 run_len = 0

#     # 시퀀스가 True로 끝난 경우 마지막 구간 처리
#     if in_run and (min_frames <= run_len <= max_frames):
#         starts.append(run_start)

#     return 1 if starts else 0


def detect_right_turn(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    rolling_seconds=2,
    right_threshold=1.0,
    duration_sec=3,
    max_duration_sec=8
):
    """
    ay만을 기반으로 우측 차선 변경 판단 (지속적 ay 변화 기반)
    - 어떤 연속 구간(run)이라도 max_duration_sec를 초과하면 0
    - 초과하지 않으면서 [duration_sec, max_duration_sec] 범위의 run이 ≥1개 있으면 1
    """
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values

    min_frames = int(duration_sec * sampling_hz)
    max_frames = int(max_duration_sec * sampling_hz)

    ay_pos = ay > right_threshold  # 임계값보다 큰 경우 -> 오른쪽 가속도(True)

    in_run = False
    run_len = 0
    has_valid = False  # [min, max] 범위 run 존재 여부

    for cond in ay_pos:
        if cond:
            if not in_run:
                in_run = True
                run_len = 1
            else:
                run_len += 1

            # 최대 초과 즉시 실패
            if run_len > max_frames:
                return 0
        else:
            if in_run:
                # run 종료 시 길이 판정
                if min_frames <= run_len <= max_frames:
                    has_valid = True
            in_run = False
            run_len = 0

    # 시퀀스가 True로 끝났다면 마지막 run도 판정
    if in_run:
        if run_len > max_frames:
            return 0
        if min_frames <= run_len <= max_frames:
            has_valid = True

    return 1 if has_valid else 0