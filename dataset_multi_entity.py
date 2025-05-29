import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.errors import EmptyDataError

from _utils.utils_path import *

MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"

def compute_azimuth_time_pandas(df, 
                                x_col='PositionX (m)', 
                                y_col='PositionY (m)',
                                z_col='PositionZ (m)',):
    """
    pandas DataFrame 기반 방위각 시계열 계산
    :param df: DataFrame with columns for x, y
    :param x_col: x 좌표 컬럼명
    :param y_col: y 좌표 컬럼명
    :return: pandas Series (방위각, 라디안 단위)
    """
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    dz = df[z_col].diff()

    #* 수평 방향 (Azimuth)
    azimuth_xy = np.arctan2(dy, dx)
    
    #* 수직 기울기 (Elevation)
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    elevation = np.arctan2(dz, horizontal_dist)
    
    df['azimuth_xy'] = azimuth_xy.interpolate(method='linear')
    df['elevation'] = elevation.interpolate(method='linear')

    return df


def compute_speed_with_vel(df,
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


class LogDataset(Dataset):

    def __init__(self, 
                 log_folder_path:str|Path|None = None, 
                 simulation_folder:str|Path|None = None):

        self.log_folder_path = LOG_FOLDER_PATH

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

        # 시나리오 폴더 안에 있는 log 파일들 처리
        # sep_log_file output -> {"MapName_ScenarioName: 
        #       {{'eventlog':'eventlog_file_name'}"
        #       {'statelog':'statelog_file_name'}"
        #       {'result':'result_file_name'}"},
        #   "same structure"
        # } 
        self.local_scenario_sets = sep_log_file(self.simulation_folder)
        self.local_scenario_sets_key_list =[* self.local_scenario_sets.keys()]

        for key, item in self.local_scenario_sets.items():
            try:
                self._check_data_and_load(self.log_folder_path / self.simulation_folder, item)
            except:
                self.local_scenario_sets_key_list.remove(key)

    def _latest_folder(self, log_root_folder):
        # file name expression -> simulation_(date)_(time)
        # data & time -> only numeric number
        log_root_folder = Path(log_root_folder)
        folders_in_log_root  = [ log_root_folder / simulation_folder for simulation_folder in os.listdir(log_root_folder) 
                                        if os.path.isdir(log_root_folder / simulation_folder)]
        
        folders_in_log_root.sort(reverse=True)
        return folders_in_log_root[0]
        
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
        return len(self.local_scenario_sets_key_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_scenario = self.local_scenario_sets_key_list[idx]
        scenario_state_log_name = self.local_scenario_sets[current_scenario]
        data = self._check_data_and_load(self.log_folder_path/self.simulation_folder , 
                                      scenario_state_log_name)
        
        #!! Data Processing will be added right below
        
        return data
    
if __name__ == '__main__':
    test_dataset = LogDataset()
    print(len(test_dataset))
    print(test_dataset[0].head())