#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os 
import csv

trajectory_type = {
    'ST':'straight',
    'RT':'right_turn',
    'LT':'left_turn',
    'UT':'U_turn',
    'LLC':'left_lane_change',
    'RLC':'right_lane_change',
    'RA':'roundabout'
}


# %%

sample_sceanrio_log_root_dir_path = Path(r'../sample_scenario_logs_250610/')

items_in_root_dir = os.listdir(sample_sceanrio_log_root_dir_path)

sample_scenario_type_dir_path = [ sample_sceanrio_log_root_dir_path /dir for dir in items_in_root_dir 
                                 if os.path.isdir(sample_sceanrio_log_root_dir_path / dir)]


total_label_data = []

for scenario_type_dir_path in sample_scenario_type_dir_path:

    
    label = [trajectory_type[key] for key in trajectory_type if key in str(scenario_type_dir_path)]
    if label == []:
        continue

    item_in_sample_scenario_type_dir = os.listdir(scenario_type_dir_path)

    event_logs = [event for event in item_in_sample_scenario_type_dir 
                if event.endswith('_eventlog.csv')]

    labeled_data = [(logs, label[0]) for logs in event_logs]
    total_label_data.extend(labeled_data)

df_total_label_data = pd.DataFrame(total_label_data, columns=['file_name', 'trajectory type'])

# %%

label_save_path = sample_sceanrio_log_root_dir_path / 'label.csv'
print(label_save_path)
df_total_label_data.to_csv(label_save_path, index=False, encoding="utf-8")

# %%
