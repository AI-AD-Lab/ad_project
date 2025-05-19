#%%
import pandas as pd
import numpy as np
from pathlib import Path
from _utils.utils_path import sep_log_file, load_data
from _utils.utils_plot import PLOTING
from datetime import datetime
import csv
from config import config

#%%
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "test_scenario_logs"                  # From src
SINGLE_SCENARIO_LOG_DATA = MORAISIM_PATH / "single_scenario_logs"       # To dst


scenario_label_dit = config['class_to_label']

def input_scenario_type():
    print("Please Enter to continue...")
    print("1: Straight")
    print("2: Left")
    print("3: Right")
    print("4: U-turn")
    print("5: Roundabout - 회전교차로")
    print("6: Lane Change - Left")
    print("7: Lane Change - Right")
    print("8: Merge - Left")
    print("9: Merge - Right")
    print("10: Discard ")
    reply = input("Please Enter Number to continue...")
    try:
        reply = int(reply)
        if reply not in scenario_label_dit.keys():
            raise ValueError("Invalid input")
    except ValueError:
        print("Invalid input, please enter a number between 1 and 10.")
        return input_scenario_type()
    return reply


def add_row(df, val_list):
    df.loc[len(df)] =  val_list
    return df


def make_dst_path():
    date_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    dst_folder_path = SINGLE_SCENARIO_LOG_DATA / f"created_{date_time}"
    dst_folder_path.mkdir(parents=True, exist_ok=True)

    return dst_folder_path

if __name__ == "__main__":

    test_path =  LOG_FOLDER_PATH / 'simulation_t9_left'
    scenario_log_names = sep_log_file(test_path)

    dst_dir_path = make_dst_path()
    dst_label_csv_path = dst_dir_path / "label.csv"

    cols = ['state_log_name', 'Label']
    tmp_df = pd.DataFrame(columns=cols)

    for key in scenario_log_names:
        log_set = scenario_log_names[key]
        state_log_data = load_data(test_path,log_set)
        
        if state_log_data.empty:
            print("state_log_data is empty")
            continue

        entity_data_dict = { entity: state_log_data[state_log_data['Entity'] == entity].reset_index(drop=True)
        for entity in pd.unique(state_log_data['Entity'])
        }

        for entity_key in entity_data_dict.keys():

            ego_position = entity_data_dict[entity_key]
            state_log_name = log_set['statelog'].replace('_statelog.csv', f'_{entity_key}.csv')

            if entity_key != "Ego":
                extra_position = ego_position.copy()
                extra_position['Entity'] = 'Ego'
                ploting_ = PLOTING(extra_position)
                ploting_.plot_trajectory(show_velocity=True, elev=30, azim=70, save=False, show=True)
            else:
                ploting_ = PLOTING(ego_position)
                ploting_.plot_trajectory(show_velocity=True, elev=30, azim=70, save=False, show=True)

            reply = input_scenario_type()
            if reply == 10:
                print("Discarding this scenario...")
                continue
                
            ego_position['Entity'] = 'Ego'
            tmp_df = add_row(tmp_df, [state_log_name,scenario_label_dit[reply]])

            file_entity = log_set['statelog'].replace('_statelog.csv', '_') + f'{entity_key}' + '.csv'
            dataframe_path = dst_dir_path / file_entity

            real_trajectory_save_path  = dst_dir_path / dataframe_path
            print(f"Scenario {scenario_label_dit[reply]} saved to {real_trajectory_save_path}")            
            ego_position.to_csv(real_trajectory_save_path)


    tmp_df.to_csv(dst_label_csv_path, index=False)
    print('Work Done!')
# %%