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
    from _utils.data_processing_utils import sample_specific_labels, TrajectoryDataset
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'
    
    confusion_matrix_save_dir = './output/plots/score/'
    # dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR)

    total_len = []

    # train_idx, test_idx = split_train_test_by_class(
    # dataset.df,
    # label_col="trajectory type",
    # test_ratio=0.3,
    # seed=42,
    # )

    # train_dataset = Subset(dataset, train_idx)
    # test_dataset  = Subset(dataset, test_idx)
    
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_data, test_data = sample_specific_labels(
        pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv'), ratio=0.3)
    train_dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR,train_data)
    test_dataset  = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR, test_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    models = {
        "rnn": RNNModel(),
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
    optimizers = {}
    schedulers = {}

    for name, model in models.items():
        optimizers[name] = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        schedulers[name] = torch.optim.lr_scheduler.OneCycleLR(
            optimizers[name], **scheduler_kwargs
        )

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
     
    for epoch in range(EPOCHS):
        # -----------------------
        #  TRAIN 단계
        # -----------------------
        for name in models.keys():
            total_loss_dict[name] = 0
            total_acc_dict[name] = 0

        total_samples = 0

        print(f"\nEpoch {epoch+1}")
        print("--- Train ---")
        for data, idx, _ in train_loader:
            ts = data.to(torch.float32).to(device)   # (B, 900, 9)
            idx = idx.to(device)
            batch_size = ts.size(0)
            total_samples += batch_size

            # 각 모델에 대해 공통 처리
            for name in models.keys():
                optimizers[name].zero_grad()

                out = models[name](ts)

                # 라벨이 1~7 → 0~6 변환
                loss = criterion(out, idx - 1)

                total_loss_dict[name] += loss.item() * batch_size
                total_acc_dict[name] += (out.argmax(dim=1) == (idx - 1)).sum().item()
                
                loss.backward()
                optimizers[name].step()
                schedulers[name].step()

        # epoch별 train 출력
        for name in models.keys():
            avg_loss = total_loss_dict[name] / total_samples
            avg_acc  = total_acc_dict[name] / total_samples
            print(f" {name}: loss - {avg_loss:.4f}, accuracy - {avg_acc:.4f}")

            train_loss_history[name].append(avg_loss)
            train_acc_history[name].append(avg_acc)

        # -----------------------
        #  TEST 단계
        # -----------------------
        print("--- Test ---")
        for name in models.keys():
            total_loss_dict[name] = 0
            total_acc_dict[name] = 0

        total_test_samples = 0

        # test는 gradient 없음
        with torch.no_grad():
            for data, idx, _ in test_loader:
                ts = data.to(torch.float32).to(device)
                idx = idx.to(device)
                batch_size = ts.size(0)
                total_test_samples += batch_size

                for name in models.keys():
                    out = models[name](ts)

                    loss = criterion(out, idx - 1)

                    total_loss_dict[name] += loss.item() * batch_size
                    total_acc_dict[name] += (out.argmax(dim=1) == (idx - 1)).sum().item()

        # epoch별 test 출력
        for name in models.keys():
            avg_loss = total_loss_dict[name] / total_test_samples
            avg_acc  = total_acc_dict[name] / total_test_samples
            print(f" {name}: loss - {avg_loss:.4f}, accuracy - {avg_acc:.4f}")
            
            test_loss_history[name].append(avg_loss)
            test_acc_history[name].append(avg_acc)
            # -----------------------
            #  최고 정확도 갱신 시 가중치 저장
            # -----------------------
            if avg_acc > best_acc[name]:
                best_acc[name] = avg_acc
                save_dir = Path("./output/model_weights/")
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)

                save_path = save_dir / f"best_{name}_model_banila_all.pt"
                torch.save(models[name].state_dict(), save_path)
                print(f"   ↳ Saved new best model for {name} (acc={avg_acc:.4f}) → {save_path}")

# %%
