#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from config import config
import itertools
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn

#%%
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH / 'simulation_TOTAL_250626'

cls_label = config['class_to_label']
label_cls = config['label_to_class']
short_to_long_label = config['Short_to_Long_Label']

label_data = pd.read_csv(SINGLE_SCENARIO_SYNLOG_DATA_ROOT / 'label.csv')
labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']
perms = list(itertools.permutations(labels))

def data_load(data_file_path):
    USEDCOLUMNS = config['data_columns']
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")
    _data = pd.read_csv(data_file_path)
    _data = _data[USEDCOLUMNS]
    return _data

labeled_data = [[] for _ in range(len(cls_label))]
# real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }


#%%
# LSTM model

embedding_dim = 1
hidden_dim = 7
num_layers = 7

model = torch.nn.LSTM(input_size=embedding_dim,
                      hidden_size=hidden_dim,
                      num_layers=num_layers,
                      batch_first=True)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



best_loss = float('inf')
best_epoch = 0
best_model = None

best_all_losses = None
best_all_preds = None
best_all_trues = None

sampling_hz = 50
rolling_window = 100

for epoch in range(10):

    all_losses = []
    all_preds  = []
    all_trues  = []

    for phase in ['train', 'test']:
        for file, label in zip(label_data['file_name'], label_data['trajectory type']):
            file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
            data = data_load(file_path)

            df_copy = data.loc[:, ~data.columns.isin(['Entity'])].copy()
            df_rolling = df_copy.rolling(rolling_window).mean().bfill()
            df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02

            # --- 입력 시계열 준비 ---
            acc_y = df_rolling['AccelerationY(EntityCoord) (m/s2)'].values.astype('float32')  # (T,)
            x = torch.from_numpy(acc_y).unsqueeze(-1)  # (T, 1)  <- feature dim 추가
            # 모델이 batch_first=True 를 기대한다고 가정하여 배치 차원 추가
            x = x.unsqueeze(0)           # (1, T, 1)

            # --- 라벨 준비 ---
            y = torch.tensor(label_cls[label]-1, dtype=torch.long, )  # 스칼라 클래스 인덱스

            if phase == "train":
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs, (hn, cn) = model(x)  # 보통 outputs: (1, T, C) 또는 (T, 1, C) 또는 (1, T, H)
                    # 마지막 타임스텝 로짓만 사용 (배치=1 가정)
                    if outputs.dim() == 3:
                        # 배치우선(1, T, C) 또는 시퀀스우선(T, 1, C) 모두 커버
                        if outputs.shape[0] == 1:       # (1, T, C) -> 마지막 타임스텝
                            logits = outputs[:, -1, :]  # (1, C)
                        else:                            # (T, 1, C)
                            logits = outputs[-1, :, :]  # (1, C)
                    else:
                        # (1, C) 같은 형태면 그대로 사용
                        logits = outputs

                    loss = criterion(logits, y.unsqueeze(0))  # (N=1,C) vs (N=1)
                    loss.backward()
                    optimizer.step()

                all_losses.append(loss.item())
                pred = torch.argmax(logits, dim=-1).item()
                all_preds.append(pred)
                all_trues.append(y.item())

            elif phase == "test":
                with torch.no_grad():
                    outputs, (hn, cn) = model(x)
                    if outputs.dim() == 3:
                        if outputs.shape[0] == 1:
                            logits = outputs[:, -1, :]
                        else:
                            logits = outputs[-1, :, :]
                    else:
                        logits = outputs

                    loss = criterion(logits, y.unsqueeze(0))
                    all_losses.append(loss.item())

                    pred = torch.argmax(logits, dim=-1).item()
                    all_preds.append(pred)
                    all_trues.append(y.item())

    # (선택) 결과 요약 출력
    if len(all_losses) > 0:
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"{phase.capitalize()} Avg Loss: {avg_loss:.4f}")

    # (선택) 정확도 같은 지표 계산
    if all_preds and all_trues:
        correct = sum(p == t for p, t in zip(all_preds, all_trues))
        acc = correct / len(all_trues)
        print(f"{phase.capitalize()} Accuracy: {acc:.3f} ({correct}/{len(all_trues)})")

    if phase == "test" and avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch
        best_model = model.state_dict()
        best_all_losses = all_losses
        best_all_preds = all_preds
        best_all_trues = all_trues

# %%
