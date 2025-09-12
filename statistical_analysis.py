#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from config import config

# %%'
MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_SYNLOG_DATA_ROOT = MORAISIM_PATH /  'simulation_TOTAL_250626'  # 시간이 일정한 데이터 파일: SYNC
label_data = pd.read_csv(SINGLE_SCENARIO_SYNLOG_DATA_ROOT / 'label.csv')

def data_load(data_file_path):
    USEDCOLUMNS = config['data_columns']
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"The file {data_file_path} does not exist.")
    _data = pd.read_csv(data_file_path)
    _data = _data[USEDCOLUMNS]
    return _data
#%%

for file, label in zip(label_data['file_name'], label_data['trajectory type']):

    file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
    data = data_load(file_path)
    described_data = data.describe()

    save_path_dir = Path(f'../FOR_REVISION/statistics')
    save_path_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_path_dir / f'{file.replace(".csv", "_described.csv")}',
    described_data.to_csv(save_path)

# %%
# 심화 분석 필요
specific_label = [
    "LRST", "RRST" , "SSST" , "ST", # STraight
    "RA3","RA9","RA12", # Roundabout
    "LT", "RT", # Turn * 2
    "UT", # U-Turn
    "LLC", "RLC" # Lane Change
]

specific_dict = {label: [] for label in specific_label}

for file, _ in zip(label_data['file_name'], label_data['trajectory type']):

    file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / file
    data = data_load(file_path)

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


# %%

tmp_specific_list = []
for key, value in specific_dict.items():
    for file in value:
        tmp_specific_list.append([key, file])

df_specific_dict = pd.DataFrame(tmp_specific_list, columns=['label', 'file_list'])
# df_specific_dict.to_csv(Path(f'../FOR_REVISION/specific_label_file_list.csv'), index=False)
# %%
df_specific_dict = None
if df_specific_dict is None:
    df_specific_dict = pd.read_csv(Path(f'../FOR_REVISION/specific_label_file_list.csv'))
df_specific_dict.head()

# %%

def data_rolling(
    df,
    sampling_hz=50,
    rolling_window=100,
):
    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz)

    return df_rolling


def confidence_interval(mean, std, confidence=0.99):
    if confidence == 0.95:
        z_score = 1.96  # 95% 신뢰구간에 해당하는 z-값
    elif confidence == 0.99:
        z_score = 2.576  # 99% 신뢰구간에 해당하는 z-값
    else:
        raise ValueError("Unsupported confidence level. Use 0.95 or 0.99.")

    margin_of_error = z_score * (std / np.sqrt(1))  # 표본 크기가 1이므로 sqrt(1) = 1
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return round(float(lower_bound), 4), round(float(upper_bound), 4)



# ST만 문리
def statistic_st(df_data):
    df_st = df_data[df_data['label'] == 'ST']
    min_list, max_list = [], []

    for item in df_st['file_list']:
        file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / item
        data = data_load(file_path)

        acc_y = data_rolling(data)['AccelerationY(EntityCoord) (m/s2)']

        # 최대 최소
        max_values = float(acc_y.max())
        min_values = float(acc_y.min())

        min_list.append(min_values)
        max_list.append(max_values)

    st_min_mean = float(np.mean(min_list))
    st_max_mean = float(np.mean(max_list))

    st_min_std = float(np.std(min_list))
    st_max_std = float(np.std(max_list))

    return st_min_mean, st_min_std, st_max_mean, st_max_std

    # 최소 99% 존재 범위 [-0.0449 -0.0141]
    # 최대 99% 존재 범위 [0.0051 0.0268]


# %%
st_min_mean, st_min_std, st_max_mean, st_max_std = statistic_st(df_specific_dict)
min_low, min_high = confidence_interval(st_min_mean, st_min_std, confidence=0.99)
max_low, max_high = confidence_interval(st_max_mean, st_max_std, confidence=0.99)

print(min_low, min_high)
print(max_low, max_high)
# %%

# ST만 문리
def statistic_ssst(df_data):
    df_st = df_data[df_data['label'] == 'RRST']
    min_list, max_list = [], []

    for item in df_data['file_list']:
        file_path = SINGLE_SCENARIO_SYNLOG_DATA_ROOT / item
        file_number = int(str(file_path).split('_')[-2])

        # if not file_number < 10:
        #     break

        data = data_load(file_path)
        acc_y = data_rolling(data)['AccelerationY(EntityCoord) (m/s2)']

        if file_number == 1:
            plt.figure(figsize=(10, 4))
            plt.plot(data['time (sec)'], acc_y.values, label='Y axis acc')
            plt.title(f'File: {item}')
            plt.xlabel('Time (sec)')
            plt.ylabel('Acceleration (m/s2)')
            # Y축 0.3에 회색 선 추가
            plt.axhline(y=0.3, color='gray', linestyle='--')
            plt.axhline(y=-0.3, color='gray', linestyle='--')
            plt.legend()
            if not os.path.exists(f'../FOR_REVISION/figures'):
                os.makedirs(f'../FOR_REVISION/figures')
            plt.savefig(f'../FOR_REVISION/figures/{item}_acc_y.png')
        else:
            continue

#         # 최대 최소
#         max_values = float(acc_y.max())
#         min_values = float(acc_y.min())

#         min_list.append(min_values)
#         max_list.append(max_values)

#     st_min_mean = float(np.mean(min_list))
#     st_max_mean = float(np.mean(max_list))

#     st_min_std = float(np.std(min_list))
#     st_max_std = float(np.std(max_list))

#     return st_min_mean, st_min_std, st_max_mean, st_max_std

# ssst_min_mean, ssst_min_std, ssst_max_mean, ssst_max_std = statistic_ssst(df_specific_dict)
# min_low, min_high = confidence_interval(ssst_min_mean, ssst_min_std, confidence=0.99)
# max_low, max_high = confidence_interval(ssst_max_mean, ssst_max_std, confidence=0.99)

# print(round(ssst_min_mean, 4), round(ssst_min_std, 4), round(min_low, 4), round(min_high, 4))
# print(round(ssst_max_mean, 4), round(ssst_max_std, 4), round(max_low, 4), round(max_high, 4))

statistic_ssst(df_specific_dict)
# %%
