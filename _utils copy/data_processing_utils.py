import pandas as pd

def normalize_time(data:pd.DataFrame) -> pd.DataFrame:
    # 인덱스 * 0.016666
    data['time (sec)'] = data.index * 0.016666
    data = data.reset_index(drop=True)
    if 'Unnamed: 0' in data.columns:
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data