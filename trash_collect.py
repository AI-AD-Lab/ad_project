import numpy as np

def extract_features(df):
    coords = df[['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']].values
    time = df['time (sec)'].values

    # 속도 및 가속도 계산
    dt = np.diff(time) + 1e-6  # 안정성 확보
    velocities = np.diff(coords, axis=0) / dt[:, None]  # (N-1)x3
    accels = np.diff(velocities, axis=0) / dt[:-1, None]  # (N-2)x3

    # 통계 기반 특징
    features = {
        'mean_accel_x': np.mean(accels[:, 0]),
        'mean_accel_y': np.mean(accels[:, 1]),
        'mean_accel_z': np.mean(accels[:, 2]),
        'std_accel_y': np.std(accels[:, 1]),
        'max_accel_y': np.max(accels[:, 1]),
        'min_accel_y': np.min(accels[:, 1])
    }
    return features

def rule_based_classifier(features):
    # 예시 룰: 가속 방향과 크기 기반 판단
    if features['max_accel_y'] > 1.5 and features['mean_accel_y'] > 0.5:
        return 'left_turn'
    elif features['min_accel_y'] < -1.5 and features['mean_accel_y'] < -0.5:
        return 'right_turn'
    elif features['mean_accel_x'] > 1.0:
        return 'acceleration'
    elif features['mean_accel_x'] < -1.0:
        return 'braking'
    else:
        return 'normal'