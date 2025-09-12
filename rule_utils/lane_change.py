from rule_utils.common_util import rolling

def detect_left_lane_change(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    threshold=0.1,
    rolling_seconds=2,
    duration_sec=0.8
):
    # data smoothing, reduce noise

    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz) # duration in frames

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

        return starting_points if starting_points else -1

    # 조건 배열
    ay_neg = (ay < -threshold) #& (ay > -1.0)
    ay_pos = (ay > threshold)

    # 최초 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if (neg_start == -1) or (pos_start == -1):
        threshold /= 2
        min_frames *= 1.2
        ay_neg = (ay < -threshold)
        ay_pos = (ay > threshold) #& (ay < 1.0)

        neg_start = find_starting_idxs(ay_neg)
        pos_start = find_starting_idxs(ay_pos)

        if (neg_start == -1) or (pos_start == -1):
            return 0

    if min(neg_start) < min(pos_start):
        return 1
    elif min(pos_start) < min(neg_start):
        return 0
    else:
        return 0

def detect_right_lane_change(
    df,
    ay_col='AccelerationY(EntityCoord) (m/s2)',
    sampling_hz=50,
    threshold=0.1,
    rolling_seconds=2,
    duration_sec=1
):
    df_rolling = rolling(df, rolling_seconds=rolling_seconds, sampling_hz=sampling_hz)
    ay = df_rolling[ay_col].values
    min_frames = int(duration_sec * sampling_hz) # duration in frames

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

        return starting_points if starting_points else -1

    # 조건 배열
    ay_neg = (ay < -threshold)
    ay_pos = (ay > threshold) #& (ay < 1.0)

    # 최초 이벤트 인덱스 탐지
    neg_start = find_starting_idxs(ay_neg)
    pos_start = find_starting_idxs(ay_pos)

    if (neg_start == -1) or (pos_start == -1):
        threshold /= 2
        min_frames *= 1.2
        ay_neg = (ay < -threshold)
        ay_pos = (ay > threshold) #& (ay < 1.0)

        neg_start = find_starting_idxs(ay_neg)
        pos_start = find_starting_idxs(ay_pos)

        if (neg_start == -1) or (pos_start == -1):
            return 0

    if min(neg_start) < min(pos_start):
        return 0
    elif min(pos_start) < min(neg_start):
        return 1
    else:
        return 0