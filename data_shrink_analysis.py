#%%
import pandas as pd
from pathlib import Path
import numpy as np

from config import config

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from _utils.plot_utils import plot_2d_graph

def shrink_ts(y, target_len=2500):
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
        vehicle_data.loc[:, position_cols] = (
            vehicle_data.loc[:, position_cols] - vehicle_data.loc[:, position_cols].iloc[0]
        )

        vehicle_data = vehicle_data.rolling(100).mean().bfill()
        vehicle_data = vehicle_data.reset_index(drop=True).to_numpy(dtype=np.float32)
        
        label = torch.tensor(config['label_to_class'][label], dtype=torch.long)
        vehicle_data = shrink_ts(vehicle_data, target_len=2500)

        return vehicle_data, label, file_name

def split_train_test_by_class(df, label_col="trajectory type", test_ratio=0.2, seed=42):
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

    # 최종적으로 한번 섞어주고 싶으면 셔플
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices



if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

    dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR)

    total_len = []

    train_idx, test_idx = split_train_test_by_class(
    dataset.df,
    label_col="trajectory type",
    test_ratio=0.3,
    seed=42,
    )

    train_dataset = Subset(dataset, train_idx)
    test_dataset  = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for batch_data, batch_labels, batch_file_names in train_loader:

        total_len.append(batch_data.shape[0])

        save_dir = Path("./output/shrink_data_2500/")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        for i in set(batch_labels.numpy()):
            class_dir = save_dir / f"class_{i}"
            if not class_dir.exists():
                class_dir.mkdir(parents=True)

        for data, label, file_name in zip(batch_data, batch_labels, batch_file_names):
            data = pd.DataFrame(data, columns=['PositionX (m)','PositionY (m)','PositionZ (m)',
                                              'VelocityX(EntityCoord) (km/h)','VelocityY(EntityCoord) (km/h)','VelocityZ(EntityCoord) (km/h)',
                                              'AccelerationX(EntityCoord) (m/s2)', 'AccelerationY(EntityCoord) (m/s2)', 'AccelerationZ(EntityCoord) (m/s2)'])

            save_path = save_dir / f"class_{label.item()}" / f"{file_name.replace('.csv', '_shrink.csv')}"
            # data.to_csv(save_path, index=False)

            data['time (sec)'] = data.index * 0.2
            save_plot_path = save_dir / f"class_{label.item()}" / f"{file_name.replace('.csv', '.png')}"
            plot_2d_graph(data, x='time (sec)', y='AccelerationY(EntityCoord) (m/s2)', legend=False, save_path=str(save_plot_path))



# %%
