import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
from config import config
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(".."))

def normalize_time(data:pd.DataFrame) -> pd.DataFrame:
    # 인덱스 * 0.016666
    data['time (sec)'] = data.index * 0.016666
    data = data.reset_index(drop=True)
    if 'Unnamed: 0' in data.columns:
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

def data_load(data_file_path):
    USEDCOLUMNS = config['data_columns']
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")
    _data = pd.read_csv(data_file_path)
    _data = _data[USEDCOLUMNS]
    return _data



def adjust_length(y, target_len=2500):
    """
    길이가 target_len보다 긴 시계열 y를 짧게 변환.
    - y: 1D array-like (numpy/pandas Series)
    - target_len: 최종 길이
    - method: 'interp' | 'avgpool' | 'downsample' | 'crop'
    - t: 시간축(초). 있다면 보간에 사용
    - fs: 샘플링 주파수(Hz). downsample 방법에서 안티앨리어싱 설정에 사용(선택)
    """

    n, d = y.shape

    # (1) 길이가 충분할 때: 마지막 target_len 개만 자르기
    if n >= target_len:
        return y[n - target_len:]     # shape: (target_len, d)

    # (2) 부족할 때: 앞쪽 0 패딩
    pad_len = target_len - n
    pad = np.zeros((pad_len, d), dtype=y.dtype)
    return np.vstack([pad, y])


class TrajectoryDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            csv_path (str or Path): label.csv 파일 경로
            data_root (str or Path): 데이터 파일들이 위치한 루트 디렉토리
            transform (callable, optional): 데이터 변환 함수 (이미지 전처리 등)
        """
        label_data = Path(data_root) / 'label.csv'
        self.df = pd.read_csv(label_data)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        label = row["trajectory type"]

        # 예시: 이미지 파일 로드
        file_path = self.data_root / file_name
        trajectory_data = pd.read_csv(file_path)
        trajectory_data = trajectory_data[trajectory_data["Entity"] == "Ego"]
        vehicle_data = trajectory_data[['PositionX (m)','PositionY (m)','PositionZ (m)',
                                            'VelocityX(EntityCoord) (km/h)','VelocityY(EntityCoord) (km/h)','VelocityZ(EntityCoord) (km/h)',
                                            'AccelerationX(EntityCoord) (m/s2)', 'AccelerationY(EntityCoord) (m/s2)', 'AccelerationZ(EntityCoord) (m/s2)'
                                            ]].copy()
        
        position_cols = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']
        if all(col in vehicle_data.columns for col in position_cols):
            vehicle_data.loc[:, position_cols] = (
                vehicle_data.loc[:, position_cols] - vehicle_data.loc[:, position_cols].iloc[0]
            )

        vehicle_data = vehicle_data.rolling(100).mean().bfill()
        vehicle_data = vehicle_data.reset_index(drop=True).to_numpy(dtype=np.float32)
        
        label = torch.tensor(config['label_to_class'][label], dtype=torch.long)
        vehicle_data = adjust_length(vehicle_data, target_len=2500)

        return vehicle_data, label, file_name
    

def split_train_test_by_class(df, label_col="trajectory type", test_ratio=0.2, seed=42):
    """
    split the dataset into train and test sets while maintaining class distribution.
    - df: pandas DataFrame containing the dataset with labels
    """
    rng = np.random.default_rng(seed)

    labels = df[label_col].values
    unique_labels = np.unique(labels)

    train_indices = []
    test_indices = []

    for lb in unique_labels:
        # 해당 클래스의 모든 인덱스
        class_idx = np.where(labels == lb)[0]
        # 셔플
        rng.shuffle(class_idx)

        n_total = len(class_idx)
        n_test = int(np.round(n_total * test_ratio))

        class_test_idx = class_idx[:n_test]
        class_train_idx = class_idx[n_test:]

        test_indices.extend(class_test_idx.tolist())
        train_indices.extend(class_train_idx.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def split_train_test_val_by_class(df, label_col="trajectory type", test_ratio=0.2, val_ratio=0.2, seed=42):
    """
    df: TrajectoryDataset.df (label.csv 로드한 DataFrame)
    label_col: 라벨이 들어있는 컬럼 이름 (여기서는 'trajectory type')
    test_ratio: 전체 중 test 비율
    return: train_indices, test_indices (둘 다 리스트)
    """
    rng = np.random.default_rng(seed)

    labels = df[label_col].values
    unique_labels = np.unique(labels)

    train_indices = []
    test_indices = []
    val_indices = []

    for lb in unique_labels:
        # 해당 클래스의 모든 인덱스
        class_idx = np.where(labels == lb)[0]
        # 셔플
        rng.shuffle(class_idx)

        n_total = len(class_idx)
        n_test = int(np.round(n_total * test_ratio))
        n_val = int(np.round(n_total * val_ratio))

        class_test_idx = class_idx[:n_test]
        class_val_idx = class_idx[n_test:n_test+n_val]
        class_train_idx = class_idx[n_test:+n_val:]


        test_indices.extend(class_test_idx.tolist())
        val_indices.extend(class_val_idx.tolist())
        train_indices.extend(class_train_idx.tolist())

    # 최종적으로 한번 섞어주고 싶으면 셔플
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices, val_indices


def sample_specific_labels(label_data_path, df_specific_dict: pd.DataFrame, ratio: float=0.3) -> pd.DataFrame:
    specific_label = [
        "LRST", "RRST" , "SSST" , "ST", # STraight
        "RA3","RA9","RA12", # Roundabout
        "LT", "RT", # Turn * 2
        "UT", # U-Turn
        "LLC", "RLC" # Lane Change
    ]
    label_data = pd.read_csv(label_data_path)
    specific_dict = {label: [] for label in specific_label}
    for file, _ in zip(label_data['file_name'], label_data['trajectory type']):
        if "ST" in file:
            if 'LRST' in file:
                specific_dict['LRST'].append(file)
            elif 'RRST' in file:
                specific_dict['RRST'].append(file)
            elif 'SSST' in file:
                specific_dict['SSST'].append(file)
            else:
                specific_dict['ST'].append(file)

        if "RA" in file:
            if 'RA3' in file:
                specific_dict['RA3'].append(file)
            elif 'RA9' in file:
                specific_dict['RA9'].append(file)
            elif 'RA12' in file:
                specific_dict['RA12'].append(file)

        if "LT" in file:
            specific_dict['LT'].append(file)
        if "RT" in file:
            specific_dict['RT'].append(file)
        if "UT" in file:
            specific_dict['UT'].append(file)
        if "LLC" in file:
            specific_dict['LLC'].append(file)
        if "RLC" in file:
            specific_dict['RLC'].append(file)

    tmp_specific_list = []
    for key, value in specific_dict.items():
        for file in value:
            tmp_specific_list.append([key, file])

    train_list = []
    test_list = []

    df_specific_dict = pd.DataFrame(tmp_specific_list, columns=['trajectory type', 'file_name'])

    # ---------------------------
    # 1) 각 세부 라벨별 0.7/0.3 나누기
    # ---------------------------
    for label in specific_label:
        df_label = df_specific_dict[df_specific_dict["trajectory type"] == label]
        if len(df_label) == 0:
            continue

        n_total = len(df_label)
        n_test = max(1, int(n_total * ratio))  # 테스트 30%
        n_train = n_total - n_test                   # 나머지 70%

        # 섞기
        df_shuffled = df_label.sample(frac=1)

        df_train = df_shuffled.iloc[:n_train]
        df_test = df_shuffled.iloc[n_train:]

        train_list.append(df_train)
        test_list.append(df_test)

    df_train_specific = pd.concat(train_list, ignore_index=True)
    df_test_specific = pd.concat(test_list, ignore_index=True)

    df_train_specific["trajectory type"] = df_train_specific["trajectory type"].map(merge_label)
    df_test_specific["trajectory type"] = df_test_specific["trajectory type"].map(merge_label)

    # 최종 DataFrame 구성
    df_train = df_train_specific[["trajectory type", "file_name"]].reset_index(drop=True)
    df_test = df_test_specific[["trajectory type", "file_name"]].reset_index(drop=True)

    return df_train, df_test


def merge_label(label: str) -> str:
    if label in ["LRST", "RRST", "SSST", "ST"]:
        return "ST"
    elif label in ["RA3", "RA9", "RA12"]:
        return "RA"
    else:
        # 나머지는 그대로 사용 (LT, RT, UT, LLC, RLC 등)
        return label
