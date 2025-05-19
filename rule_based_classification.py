#%%
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch

# %%
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
'''
used_columns: Index(['PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'RotationZ (deg)',
       'FrontWheelAngle (deg)', 'speed', 'acc']
'''
path  = r"C:\Users\tndah\Desktop\STAGE_WS\single_scenario_logs\created_20250519155704\20250519_134333_R_KR_PG_K-City_tcar_t2_9_left_Ego.csv"
data = pd.read_csv(path)

#%%
# 시간에 따른 FrontWheelAngle 시각화
import matplotlib.pyplot as plt
from _utils.data_processing_utils import normalize_time
from _utils.utils_plot import time_base_plot
from config

MORAISIM_PATH = Path(__file__).resolve().parent.parent
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

# %%
import os
all_single_entity_file = os.listdir(SINGLE_SCENARIO_LOG_DATA / 'created_20250428150508')
all_single_entity_file = [file for file in all_single_entity_file if not file.endswith('label.csv')]
# %%
for file in all_single_entity_file:
    file_path = SINGLE_SCENARIO_LOG_DATA/ 'created_20250428150508' / file
    data = pd.read_csv(file_path)
    data = normalize_time(data)


    time_base_plot(data, save_path=f"./output2/wheel/{file.replace('.csv', '')}_plot.png")
    # label = rule_based_classifier(ex_data)

    # CSV 파일에 레이블 추가
    # label_df = pd.DataFrame({'Label': [label]})
    # label_df.to_csv(file_path, mode='a', header=False, index=False)