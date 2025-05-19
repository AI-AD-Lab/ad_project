from pathlib import Path
import pandas as pd
import os
from pandas.errors import EmptyDataError

MORAISIM_PATH = Path(__file__).resolve().parent.parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"

def get_filename(path):
    return Path(path).name

def remove_numeric_parts(filename: str) -> str:
    parts = filename.split('_')[2:]
    return '_'.join(parts)

def sep_log_file(log_folder_path):
    log_name_dict = {}

    log_files = os.listdir(log_folder_path)
    statelog_base_names = [ statelog.rstrip('_eventlog.csv') 
                          for statelog in log_files 
                          if statelog.endswith('eventlog.csv')]

    for base_name in statelog_base_names:
        log_files_map = {
            "statelog": None,
            "eventlog": None,
            "result": None
        }

        base_name_logs = [logs for logs in log_files if base_name in logs]
        for file in base_name_logs:
            for key in log_files_map:
                extension = key + '.csv'
                if file.endswith(extension):
                    log_files_map[key] = file
                    break
        
        log_name_dict[base_name] = log_files_map
        
    return log_name_dict


def load_data(log_folder_path , file_dict):
    state_log = file_dict['statelog']
    event_log = file_dict['eventlog']
    result_log = file_dict['result']

    try:
        data =  pd.read_csv(log_folder_path / state_log)
        if data.empty:
            print("DATAEMPTY")
            raise EmptyDataError('No data')
        return data
    
    except EmptyDataError:
        print(f"❌ {state_log}: statelog - 빈 파일 (헤더도 없음) → 건너뜀")

    
    return state_log

