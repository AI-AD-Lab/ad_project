#%%
import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn.functional as F

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.errors import EmptyDataError

from _utils.utils_path import *

from config import config


#%%
import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
import torch.nn.functional as F

#%%
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "test_scenario_logs"                  # From src
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst

def compute_vel(df,
                            x_col='VelocityX(EntityCoord) (km/h)', 
                            y_col='VelocityY(EntityCoord) (km/h)', 
                            z_col='VelocityZ(EntityCoord) (km/h)',
                           ):
    """
    pandas DataFrame 기반 속도 시계열 계산
    :param df: DataFrame with columns for x, y, timestamp
    :param x_col: x기준 속도 컬럼명
    :param y_col: y기준 속도 컬럼명, 단 z는 불필요하다 생각(현재)
    :param time_col: 시간 컬럼명 (sec)
    :return: pandas Series (speed per time step)
    """
    dx = df[x_col]
    dy = df[y_col]
    dz = df[z_col]

    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    df['speed'] = speed.bfill()
    return df

def compute_acc(df,
                            x_col='AccelerationX(EntityCoord) (m/s2)', 
                            y_col='AccelerationY(EntityCoord) (m/s2)', 
                            z_col='AccelerationZ(EntityCoord) (m/s2)',
                           ):
    dx, dy, dz = df[x_col],  df[y_col],  df[z_col]
    acc = np.sqrt(dx**2 + dy**2 + dz**2)
    df['acc'] = acc.bfill()
    return df
    
class LogDataset(Dataset):

    def __init__(self, 
                 log_folder_path:str|Path|None = None, 
                 simulation_folder:str|Path|None = None):

        self.log_folder_path = SINGLE_SCENARIO_LOG_DATA
        self.config = config['label_to_class']

        if log_folder_path:
            self.log_folder_path = log_folder_path
        
        self.log_folder_path = Path(self.log_folder_path)
        if not self.log_folder_path.exists():
            raise FileNotFoundError(f"Log folder {self.log_folder_path} does not exist.")
        
        # MORAISIM_PATH / "logs_scenario_runner" / ** -> dir
        if simulation_folder:
            self.simulation_folder = simulation_folder
        else:
            self.simulation_folder = self._latest_folder(self.log_folder_path)


        self.simulation_folder_path = SINGLE_SCENARIO_LOG_DATA / self.simulation_folder

        self.label_csv = 'label.csv'
        self.state_log_csvs = [csv_file for csv_file in os.listdir(self.simulation_folder_path) \
                                if csv_file.endswith('.csv')] 
        self.state_log_csvs.remove(self.label_csv)

        label_path = self.simulation_folder_path /self.label_csv
        self.label_data = pd.read_csv(label_path)
        self.label_data['Label'] = self.label_data['Label'].apply(self.change_label_to_class)

    def change_label_to_class(self,x):
        return self.config[x]

    def _latest_folder(self, log_root_folder):
        # file name expression -> simulation_(date)_(time)
        # data & time -> only numeric number
        folders = [f for f in os.listdir(self.log_folder_path) if Path.is_dir(self.log_folder_path / f)]
        folders.sort(reverse=True)
        return folders[0]
        
    def _check_data_and_load(self, folder_path , dictionary_data):
        state_log_name = dictionary_data['statelog']
        folder_path = Path(folder_path)
        try:
            data =  pd.read_csv(folder_path / state_log_name)
            if data.empty:
                raise EmptyDataError("DataFrame is empty")
            return data
        except EmptyDataError:
            print(f"❌ {state_log_name}: statelog - No Header → 건너뜀")

    def __len__(self):
        return self.label_data.shape[0]
    
    def __getitem__(self, idx):
        statelog_file_name =  self.label_data['state_log_name'][idx]
        label = self.label_data['Label'][idx]

        statelog_file_path = self.simulation_folder_path / statelog_file_name

        statelog_data = pd.read_csv(statelog_file_path)
        statelog_data = statelog_data[config['data_columns']]
        statelog_data = compute_vel(statelog_data)
        statelog_data = compute_acc(statelog_data)
        statelog_data = statelog_data.drop(columns=[
            'time (sec)',
            'VelocityX(EntityCoord) (km/h)',
            'VelocityY(EntityCoord) (km/h)', 
            'VelocityZ(EntityCoord) (km/h)', 
            'AccelerationX(EntityCoord) (m/s2)',
            'AccelerationY(EntityCoord) (m/s2)',
            'AccelerationZ(EntityCoord) (m/s2)',
        ])
        statelog_data = statelog_data.fillna(0)
        print(f'used_columns: {statelog_data.columns}')
        data_tensor = torch.tensor(statelog_data.values, dtype=torch.float32)
        
        return data_tensor, label
    
def pad_or_trim_2d(tensor, max_length=3000):
    seq_len = tensor.size()[0]
    if seq_len < max_length:
        # (left, right) 패딩인데, feature_dim은 유지하고 seq_len 쪽만 오른쪽에 패딩
        padding = (0, 0, 0, max_length - seq_len)  # (왼, 오, 위, 아래) 순서
        return F.pad(tensor, padding)
    else:
        return tensor[:max_length, :]  # 앞 10개 시퀀스만 자르기

def collate_fn_fixed_length(batch):
    # batch: list of 1D tensors

    data_list, label_list = zip(*batch)
    processed_data = [pad_or_trim_2d(t) for t in data_list]
    return torch.stack(processed_data), torch.tensor(label_list)

def collate_fn_variable_length(batch):
    # batch: list of 1D tensors

    data_list = torch.stack([data for data, _ in batch])
    label_list = torch.tensor([label for _, label in batch], dtype=torch.long)
    return data_list, label_list




if __name__ == '__main__':


    test_dataset = LogDataset()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=collate_fn_variable_length )
    max_len = []
    for data, label in loader:
        print(data.shape, label.shape)
        max_len.append(data.shape[1])

    