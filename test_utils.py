#%%
import pandas as pd

from _utils.utils_path import *
from _utils.utils_plot import *
from scenario_dataset import *
# %%
# |- root
#   |- logs_scenario_runner
#      |- simulation_** 
#   |- analyze_log_ws
#      |- main.py

MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"

def replace_outliers_with_nan(series, threshold=0.05):
    series = series.copy()
    delta = series.diff().abs()
    series[delta < threshold] = np.nan
    return series

def smooth_with_interpolation(series):
    """
    이상치 제거 + 선형 보간
    """
    cleaned = replace_outliers_with_nan(series)
    interpolated = cleaned.interpolate(method='linear', limit_direction='both')
    return interpolated


if __name__ == '__main__':

    test_path = LOG_FOLDER_PATH /'simulation_20250417_183629'
    dd = sep_log_file(test_path)

    for key in dd:
        log_set = dd[key]
        state_log_data = load_data(test_path,log_set)
        
        if state_log_data.empty:
            print("state_log_data is empty")

        else:
            break
    

    entitiy_data_dict = { entity: state_log_data[state_log_data['Entity'] == entity].reset_index(drop=True)
    for entity in pd.unique(state_log_data['Entity'])
    }

    ego_position = entitiy_data_dict['Ego'][['time (sec)','PositionX (m)', 'PositionY (m)', 'PositionZ (m)', 'VelocityX(EntityCoord) (km/h)',
       'VelocityY(EntityCoord) (km/h)', 'VelocityZ(EntityCoord) (km/h)', 'RotationX (deg)',
       'RotationY (deg)', 'RotationZ (deg)', 'FrontWheelAngle (deg)' ]]

    # print(ego_position['time (sec)'].diff().replace(0, np.nan).mean())


    ego_data = ego_position.copy()
    # print(ego_data['speed'].head(5))
    dataframe_2d_plot(ego_data, col1='PositionX (m)', col2='PositionY (m)')

    ego_data['new_speed'] = compute_speed_with_vel(ego_position)
    print(ego_data[['new_speed']].head(5))
    dataframe_2d_plot(ego_data, col1='time (sec)', col2='new_speed')

    gegege = ego_data[['time (sec)', 'PositionX (m)', 'PositionY (m)', 'PositionZ (m)']].copy()
    gegege = compute_azimuth_time_pandas(gegege)

    # gegege['azimuth_xy'] = smooth_column(gegege, 'azimuth_xy', window=5)
    # gegege['elevation'] = smooth_column(gegege, 'elevation', window=100)
    # gegege['speed'] = smooth_column(gegege, 'speed', window=3)

    # speed_plot(gegege, col1='time (sec)', col2='new_speed')
    dataframe_2d_plot(gegege, col1='time (sec)', col2='azimuth_xy')
    dataframe_2d_plot(gegege, col1='time (sec)', col2='elevation')

# %%
