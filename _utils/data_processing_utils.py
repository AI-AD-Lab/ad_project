import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from config import config

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



