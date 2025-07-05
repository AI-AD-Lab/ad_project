
#%%
from pathlib import Path
from _utils.utils_plot3 import plot_driving_trajectory
from _utils.utils_plot import draw_acceleration_y_plot
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# main.py 기준으로 상위 디렉터리 && target_folder 내부 파일 - target_folder: '../logs_scenario_runner'
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"
POSITIONS = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']
VELOCITY = {'x': 'VelocityX(EntityCoord) (km/h)',
            'y': 'VelocityY(EntityCoord) (km/h)',
            'z': 'VelocityZ(EntityCoord) (km/h)'}


if __name__=='__main__':
    import os
    example_dir_path = r"D:/3세부 Tier4/scenario_ws/representive_logs/"
    statelog_csv = [statelog for statelog in os.listdir(example_dir_path) if statelog.endswith('statelog.csv')]

    for statelog in statelog_csv:

        example_file_path = example_dir_path + statelog
        example_df = pd.read_csv(example_file_path)

        trajectory_plot_name = statelog.replace('statelog.csv', '_trajectory_plot.png')
        y_axis_acceleration_plot_name = statelog.replace('statelog.csv', '_y_axis_acceleration_plot.png')

        # Plotting the driving trajectory, visualizing velocity and saving the plot
        trajectory_saving_path = example_dir_path + trajectory_plot_name
        # plot_driving_trajectory(example_df, save_path=trajectory_saving_path , visualize_velocity=True)

        # Plotting the y-axis acceleration and saving the plot
        y_axis_acceleration_saving_path = example_dir_path + y_axis_acceleration_plot_name
        draw_acceleration_y_plot(example_df, save_path=y_axis_acceleration_saving_path)
# %%
