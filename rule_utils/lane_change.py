import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_wz_from_xyz(df: pd.DataFrame):
    """
    x, y 좌표 기반으로 lateral acceleration 'wz' 계산 // 
    Returns:
    - df: ay, wz가 추가된 DataFrame
    """

    # 1. 속도 계산 (v = dx/dt)
    df['time (sec)'] = df.index * 0.016666
    dt = 1.0 / 60


    vx = df['VelocityX(EntityCoord) (km/h)'] * 1000 / 3600  # m/s
    vy = df['VelocityY(EntityCoord) (km/h)'] * 1000 / 3600  # m/s

    heading = np.arctan2(vy, vx)            # 차량 진행 방향
    heading_unwrapped = np.unwrap(heading)  # 불연속 보정
    wz = np.diff(heading_unwrapped, prepend=heading_unwrapped[0]) / dt      # yaw rate (rad/s)

    df['wz'] = wz
    return df


def detect_LLC_by_ay_wz(df, ay_col='AccelerationY(EntityCoord) (m/s2)', wz_col='wz', sampling_hz=60,
                        ay_threshold=-0.5, wz_max_limit=0.1, duration_thresh=30):
    """
    ay/wz 기반 왼쪽 차선 변경(LLC) 감지 함수

    Parameters:
    - df: DataFrame containing 'ay' and 'wz'
    - sampling_hz: Hz 단위 (default: 60)
    - ay_threshold: ay가 이보다 작으면 좌측 변경 가능성
    - wz_max_limit: yaw rate가 이보다 크면 회전으로 간주 → 차선 변경 아님
    - duration_thresh: ay < ay_threshold 지속 프레임 수 기준 (default: 30 프레임 ≈ 0.5초)

    Returns:
    - True if LLC detected
    """

    df = compute_wz_from_xyz(df)

    ay = df[ay_col]
    wz = df[wz_col]

    # 조건 1: ay가 threshold보다 작게 내려간 프레임 수
    active_ay = (ay < ay_threshold).astype(int)
    max_seq = 0
    count = 0
    for val in active_ay:
        if val:
            count += 1
            max_seq = max(max_seq, count)
        else:
            count = 0

    # 조건 2: yaw rate가 너무 크지 않음 (회전이 아님)
    wz_max = wz.abs().max()

    if max_seq >= duration_thresh and wz_max < wz_max_limit:
        print(f"✅ LLC 감지됨 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
        return True

    print(f"❌ LLC 아님 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
    return False


def detect_RLC_by_ay_wz(df, ay_col='AccelerationY(EntityCoord) (m/s2)', wz_col='wz', sampling_hz=60,
                        ay_threshold=0.5, wz_max_limit=-0.1, duration_thresh=30):
    """
    ay/wz 기반 왼쪽 차선 변경(LLC) 감지 함수

    Parameters:
    - df: DataFrame containing 'ay' and 'wz'
    - sampling_hz: Hz 단위 (default: 60)
    - ay_threshold: ay가 이보다 작으면 좌측 변경 가능성
    - wz_max_limit: yaw rate가 이보다 크면 회전으로 간주 → 차선 변경 아님
    - duration_thresh: ay < ay_threshold 지속 프레임 수 기준 (default: 30 프레임 ≈ 0.5초)

    Returns:
    - True if LLC detected
    """

    df = compute_wz_from_xyz(df)

    ay = df[ay_col]
    wz = df[wz_col]

    # 조건 1: ay가 threshold보다 올라간 프레임 수
    active_ay = (ay > ay_threshold).astype(int)
    max_seq = 0
    count = 0
    for val in active_ay:
        if val:
            count += 1
            max_seq = max(max_seq, count)
        else:
            count = 0

    # 조건 2: yaw rate가 너무 크지 않음 (회전이 아님)
    wz_max = wz.abs().max()

    if max_seq <= duration_thresh and wz_max > wz_max_limit:
        print(f"✅ RLC 감지됨 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
        return True

    print(f"❌ RLC 아님 | ay 지속 프레임: {max_seq}, wz 최대값: {wz_max:.4f}")
    return False