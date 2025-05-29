#%%
import pandas as pd

from _utils.utils_path import *
from _utils.utils_plot import *
from dataset_multi_entity import *
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

    # simulation_20250421_142800  - R_KR_PR_Sangam_NoBuildings_tcar_t2_1
    test_path =  Path('../test_scenario_logs') /'simulation_20250421_142908'
    dd = sep_log_file(test_path)

    for key in dd:
        log_set = dd[key]
        state_log_data = load_data(test_path,log_set)
        
        print(f"log_set: {key}")

        if state_log_data.empty:
            print("state_log_data is empty")
            continue

        entity_data_dict = { entity: state_log_data[state_log_data['Entity'] == entity].reset_index(drop=True)
        for entity in pd.unique(state_log_data['Entity'])
        }
        ego_position = entity_data_dict['Ego']
        dada = PLOTING(state_log_data)
        dada.plot_trajectory(show_velocity=True, elev=30, azim=70, save=False, show=True)

#%%
    key = 'R_KR_PG_K-City_tcar_t2_17_clock3'
    log_set = dd[key]
    state_log_data = load_data(test_path,log_set)
    entity_data_dict = { entity: state_log_data[state_log_data['Entity'] == entity].reset_index(drop=True)
    for entity in pd.unique(state_log_data['Entity'])
    }
    # print(entity_data_dict['Ego'].keys())
    ego_position = entity_data_dict['npc_1']
    dada = PLOTING(state_log_data)
    dada.plot_trajectory(show_velocity=True, elev=30, azim=-85, save=False, show=True)


#%%

    ego_position = compute_speed_with_vel(ego_position)

    ego_position_z = compute_azimuth_time_pandas(ego_position)
    dataframe_2d_plot(ego_position_z, col1='time (sec)', col2= 'RotationZ (deg)')
    # dataframe_2d_plot(ego_position_z, col1='time (sec)', col2= 'azimuth_xy')
    # dataframe_2d_plot(ego_position_z, col1='time (sec)', col2= 'FrontWheelAngle (deg)')
    # print(ego_position_z[['RotationZ (deg)', 'azimuth_xy' ]].corr())


    # for ch in ['X', 'Y', 'Z']:
    #     print('-'*50)
    #     dataframe_2d_plot(ego_position, col1='time (sec)', col2=f'Position{ch} (m)')
    #     dataframe_2d_plot(ego_position, col1='time (sec)', col2=f'Velocity{ch}(EntityCoord) (km/h)')
    #     dataframe_2d_plot(ego_position, col1='time (sec)', col2=f'Acceleration{ch}(EntityCoord) (m/s2)')
        # dataframe_2d_plot(ego_position, col1='time (sec)', col2=f'Rotation{ch} (deg)')

    # dataframe_2d_plot(ego_position, col1='time (sec)', col2=f'speed')
    # dataframe_2d_plot(ego_position, col1='time (sec)', col2= 'Throttle [0..1]')
    # dataframe_2d_plot(ego_position, col1='time (sec)', col2= 'Brake [0..1]')


    # dataframe_2d_plot(ego_position_X, col1='time (sec)', col2= 'RotationX (deg)')
    # dataframe_2d_plot(ego_position_X, col1='time (sec)', col2= 'azimuth_xy')



    # dataframe_2d_plot(ego_position, col1='time (sec)', col2= 'elevation')


    ego_data = ego_position.copy()




# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 DataFrame (실제 df로 교체하세요)
# df = pd.read_csv("your_data.csv")  # 데이터 불러오는 예시

# 상관계수 계산
corr = ego_data.drop('Entity', axis=1).corr()

ex1 = ego_data[[
                'VelocityX(EntityCoord) (km/h)',
                'VelocityY(EntityCoord) (km/h)',
                'VelocityZ(EntityCoord) (km/h)',
                'AccelerationX(EntityCoord) (m/s2)',
                'AccelerationY(EntityCoord) (m/s2)',
                'AccelerationZ(EntityCoord) (m/s2)']]
corr1 = ex1.corr()


ex2  = ego_data[[
                'PositionX (m)',
                'PositionY (m)',
                'PositionZ (m)',
                'RotationX (deg)',
                'RotationY (deg)',
                'RotationZ (deg)']]
corr2 = ex2.corr()

ex3 = ego_data[[
                'Throttle [0..1]',
                'Brake [0..1]',
                'speed',
                'azimuth_xy',
                'elevation']]
corr3 = ex3.corr()

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(corr3, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)

# 저장
strings = 'throt - break'
plt.title(f"Correlation Heatmap {strings}")
plt.tight_layout()
plt.savefig(f"./output2/correlation_heatmap {strings}.png", dpi=300)
plt.close()


# %%
