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

USE_COLUMNS = [
    # 'PositionX (m)',
    # 'PositionY (m)',
    # 'PositionZ (m)',
    # 'VelocityX(EntityCoord) (km/h)',
    # 'VelocityY(EntityCoord) (km/h)',
    # 'VelocityZ(EntityCoord) (km/h)',
    # 'AccelerationX(EntityCoord) (m/s2)',
    'AccelerationY(EntityCoord) (m/s2)',
    # 'AccelerationZ(EntityCoord) (m/s2)'
]
INPUT_DIM = len(USE_COLUMNS)
HIDDEN_DIM = 128
OUTPUT_DIM = 7   # 분류면 클래스 개수로 변경
EPOCHS = 50
SEQ_LEN = 9076

def shrink_ts(y, target_len=900):
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

def merge_label(label: str) -> str:
    if label in ["LRST", "RRST", "SSST", "ST"]:
        return "ST"
    elif label in ["RA3", "RA9", "RA12"]:
        return "RA"
    else:
        # 나머지는 그대로 사용 (LT, RT, UT, LLC, RLC 등)
        return label

def sample_specific_labels(df_specific_dict: pd.DataFrame, ratio: float=0.3) -> pd.DataFrame:
    specific_label = [
        "LRST", "RRST" , "SSST" , "ST", # STraight
        "RA3","RA9","RA12", # Roundabout
        "LT", "RT", # Turn * 2
        "UT", # U-Turn
        "LLC", "RLC" # Lane Change
    ]
    label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
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


# GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
# SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'
# label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')

# train, test = sample_specific_labels(label_data, ratio=0.3)
# print(len(train), len(test))
# print("Specific label sampling...")

class TrajectoryDataset_spec(Dataset):
    def __init__(self, data_root, df_data, transform=None):
        """
        Args:
            csv_path (str or Path): label.csv 파일 경로
            data_root (str or Path): 데이터 파일들이 위치한 루트 디렉토리
            transform (callable, optional): 데이터 변환 함수 (이미지 전처리 등)
        """

        self.df_data = df_data
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df_data)


    def __getitem__(self, idx):
        row = self.df_data.iloc[idx]
        file_name = row["file_name"]
        label = row["trajectory type"]

        # 예시: 이미지 파일 로드
        file_path = self.data_root / file_name
        trajectory_data = pd.read_csv(file_path)
        trajectory_data = trajectory_data[trajectory_data["Entity"] == "Ego"]
        vehicle_data = trajectory_data[USE_COLUMNS].copy()
        
        position_cols = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']
        if all(col in vehicle_data.columns for col in position_cols):
            vehicle_data.loc[:, position_cols] = (
                vehicle_data.loc[:, position_cols] - vehicle_data.loc[:, position_cols].iloc[0]
            )

        vehicle_data = vehicle_data.rolling(100).mean().bfill()
        vehicle_data = vehicle_data.reset_index(drop=True).to_numpy(dtype=np.float32)

        short_to_long = config["Short_to_Long_Label"][label]
        label = torch.tensor(config['label_to_class'][short_to_long], dtype=torch.long)

        return vehicle_data, label, file_name


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
        vehicle_data = trajectory_data[USE_COLUMNS].copy()
        
        position_cols = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']
        if all(col in vehicle_data.columns for col in position_cols):
            vehicle_data.loc[:, position_cols] = (
                vehicle_data.loc[:, position_cols] - vehicle_data.loc[:, position_cols].iloc[0]
            )

        vehicle_data = vehicle_data.rolling(100).mean().bfill()
        vehicle_data = vehicle_data.reset_index(drop=True).to_numpy(dtype=np.float32)
        
        label = torch.tensor(config['label_to_class'][label], dtype=torch.long)
        # vehicle_data = shrink_ts(vehicle_data, target_len=SEQ_LEN) # 2500
        # vehicle_data = expend_ts(vehicle_data, target_len=SEQ_LEN) # 9076

        return vehicle_data, label, file_name

def split_train_test_by_class(df, label_col="trajectory type", test_ratio=0.2, seed=42):
    """
    df: TrajectoryDataset.df (label.csv 로드한 DataFrame)
    label_col: 라벨이 들어있는 컬럼 이름 (여기서는 'trajectory type')
    test_ratio: 전체 중 test 비율
    return: train_indices, test_indices (둘 다 리스트)
    """
    # rng = np.random.default_rng(seed)
    rng = np.random.default_rng()

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



# 1. 기본 RNN 모델 -------------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, 900, 9)
        out, h_n = self.rnn(x)      # out: (B, 900, H)
        last = out[:,-1, :]        # 마지막 time step 사용
        y = self.fc(last)           # (B, output_dim)
        return y


# 2. LSTM 모델 -----------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, 900, 9)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, 900, H)
        last = out[:, -1, :]            # 마지막 time step
        y = self.fc(last)               # (B, output_dim)
        return y


# 3. Transformer Encoder 모델 -------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        output_dim=OUTPUT_DIM,
    ):
        super().__init__()
        # 입력 9차원을 d_model 차원으로 projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=SEQ_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [CLS] 토큰 없이 전체 평균 pooling 사용
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (B, 900, 9)
        x = self.input_proj(x)           # (B, 900, d_model)
        x = self.pos_encoder(x)
        enc_out = self.encoder(x)        # (B, 900, d_model)

        # time dimension 평균 (또는 마지막 step, max pooling 등으로 변경 가능)
        pooled = enc_out.mean(dim=1)     # (B, d_model)
        y = self.fc(pooled)              # (B, output_dim)
        return y



if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'
    
    confusion_matrix_save_dir = './output/plots/score/'

    total_len = []


    total_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
    
    train_data, test_data = sample_specific_labels(
        total_data, ratio=0.3)
    
    train_dataset = TrajectoryDataset_spec(SYN_LOG_DATA_ROOT_DIR, train_data)
    test_dataset  = TrajectoryDataset_spec(SYN_LOG_DATA_ROOT_DIR, test_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR, total_data)
    total_loader  = DataLoader(total_dataset, batch_size=1, shuffle=False)
    
    models = {
        # "rnn": RNNModel(),
        "lstm": LSTMModel(),
        # "trans": TransformerModel()
    }
    
    # 2) Optimizer & Scheduler 설정 공통값
    optimizer_kwargs = {
        "lr": 1e-3,
        "weight_decay": 5e-4
    }

    scheduler_kwargs = {
        "max_lr": 3e-3,
        "steps_per_epoch": len(train_dataset),
        "epochs": EPOCHS,
        "pct_start": 0.3,
    }

    # 3) Optimizer/Scheduler 생성 자동화 (딱 5줄)
    # optimizers = {}
    # schedulers = {}

    # for name, model in models.items():
    #     optimizers[name] = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    #     schedulers[name] = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizers[name], **scheduler_kwargs
    #     )

    model_weight_dir = Path("./output/model_weights/")
    for name in models.keys():
        weight_path = model_weight_dir / f"best_{name}_model_banila_all.pt"
        models[name].load_state_dict(torch.load(weight_path))
        print(f"Loaded weights for {name} from {weight_path}")
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model in models.values():
        model = model.to(device)

    total_loss_dict = {name: 0.0 for name in models.keys()}
    total_acc_dict = {name: 0.0 for name in models.keys()}
    best_acc = {name: 0.0 for name in models.keys()}

    train_loss_history = {name: [] for name in models.keys()}
    train_acc_history = {name: [] for name in models.keys()}

    test_loss_history = {name: [] for name in models.keys()}
    test_acc_history = {name: [] for name in models.keys()}
     
    from time import time
    start_time = time()

    # -----------------------
    #  TEST 단계
    # -----------------------
    print("--- Test ---")
    for name in models.keys():
        total_loss_dict[name] = 0
        total_acc_dict[name] = 0

    total_test_samples = 0

    pred_dict = {name: [] for name in models.keys()}
    true_dict = {name: [] for name in models.keys()}

    # test는 gradient 없음
    with torch.no_grad():
        for data, idx, _ in total_loader:
            ts = data.to(torch.float32).to(device)
            idx = idx.to(device)
            batch_size = ts.size(0)
            total_test_samples += batch_size

            for name in models.keys():
                out = models[name](ts)

                loss = criterion(out, idx - 1)

                total_loss_dict[name] += loss.item() * batch_size
                total_acc_dict[name] += (out.argmax(dim=1) == (idx - 1)).sum().item()

                pred_dict[name].extend(out.cpu().tolist())
                true_dict[name].extend((idx-1).cpu().tolist())

    end_time = time()
    print(f"Test Time: {end_time - start_time:.2f} seconds")
    print("total test samples: {}".format(total_test_samples))
    print("processing time per sample: {:.4f} seconds".format((end_time - start_time) / total_test_samples))
    print("samples per second: {:.2f} samples/second".format(total_test_samples / (end_time - start_time)))
    # epoch별 test 출력
    f1_score_dict = {}
    from sklearn.metrics import f1_score
    for name in models.keys():
        avg_loss = total_loss_dict[name] / total_test_samples
        avg_acc  = total_acc_dict[name] / total_test_samples

        f1 = f1_score(true_dict[name], np.array(pred_dict[name]).argmax(axis=1), average='weighted')
        print(f" {name}: loss - {avg_loss:.4f}, accuracy - {avg_acc:.4f}")
        print(f" {name}: F1 Score - {f1:.4f}")
        # test_loss_history[name].append(avg_loss)
        # test_acc_history[name].append(avg_acc)

# %%
