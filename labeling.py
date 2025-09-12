#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os
import csv
from config import config

trajectory_type = config['Short_to_Long_Label']

def get_trajectory_type(file_name):
    """
    Extracts the trajectory type from the directory path.
    """
    for key in trajectory_type.keys():
        if key in str(file_name):
            return trajectory_type[key]
    return None

def get_statelog_list(dir_path):
    items = os.listdir(dir_path)
    items = [item for item in items if item.endswith('_statelog.csv')]
    return items

def make_label_csv(dir_path:Path|str|None=None, save_path:Path|str|None=None):

    if dir_path is None:
        raise ValueError("dir_path must be provided")
    if save_path is None:
        raise ValueError("save_path must be provided")

    event_log_files = get_statelog_list(dir_path)

    total_label_data = []
    for file in event_log_files:
        label = get_trajectory_type(file)

        if label is None:
            print(f"Warning: No trajectory type found for file '{file}'")
            continue

        total_label_data.append((file, label))

    df_total_label_data = pd.DataFrame(total_label_data, columns=['file_name', 'trajectory type'])
    df_total_label_data.to_csv(save_path, index=False, encoding="utf-8")

    return df_total_label_data

if __name__ == '__main__':
    # Example usage
    sample_scenario_log_root_dir_path = Path(r'../simulation_TOTAL_250626/')
    label_save_path = sample_scenario_log_root_dir_path / 'label.csv'

    make_label_csv(dir_path=sample_scenario_log_root_dir_path, save_path=label_save_path)
    print(f"Labels saved to {label_save_path}")
