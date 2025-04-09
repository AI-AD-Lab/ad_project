
#%%
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import colors as mcolors
import argparse
from pandas.errors import EmptyDataError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# main.py 기준으로 상위 디렉터리 && target_folder 내부 파일 - target_folder: '../logs_scenario_runner'
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"
POSITIONS = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']
VELOCITY = {'x': 'VelocityX(EntityCoord) (km/h)', 
            'y': 'VelocityY(EntityCoord) (km/h)', 
            'z': 'VelocityZ(EntityCoord) (km/h)'}

def sep_log_file(log_folder_path):
    '''
    data structure:
        return {
            "statelog" : *_statelog.csv,
            "evenvtlog": *_evenvtlog.csv,
            "result"   : *_result.csv
        }
    
    ../logs_scenario_runner에 있는 1개의 시나리오가 실행 된 이후 생성되는 각 개별 폴더가 인자
    결과값은 각 시나리오 로드 폴더 안 3가지 csv 파일에 대한 파일명 반환
        
    '''
    
    log_files = os.listdir(log_folder_path)

    log_files_map = {
        "statelog": None,
        "eventlog": None,
        "result": None
    }

    for file in log_files:
        for key in log_files_map:
            extension = key + '.csv'
            if file.endswith(extension):
                log_files_map[key] = file
                break

    return log_files_map

def blend_colors(base_color, blend_with, alpha=0.5):
    return tuple([
        base_color[i] * (1 - alpha) + blend_with[i] * alpha
        for i in range(3)
    ])

def sqrt_square_sum( x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    return np.sqrt(x**2 + y**2 + z**2)

def create_colored_line_segments(x, y, z, velocity, cmap='plasma', norm=None):
    x, y, z, velocity = map(np.asarray, (x, y, z, velocity))
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if norm is None:
        norm = Normalize(vmin=velocity.min(), vmax=velocity.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(velocity[:-1])
    lc.set_linewidth(2)
    return lc

def plot_3d_line_equal_unit(state_log_df, elev=30, azim=70, save=False, fig_name:str|None=None):
    '''
    state_log_df coloumns: 
            ['time (sec)', 
            'Entity',                               * REQUIRED
            'PositionX (m)',                        * REQUIRED
            'PositionY (m)',                        * REQUIRED
            'PositionZ (m)',                        * REQUIRED
            'VelocityX(EntityCoord) (km/h)', 
            'VelocityY(EntityCoord) (km/h)', 
            'VelocityZ(EntityCoord) (km/h)', 
            'AccelerationX(EntityCoord) (m/s2)', 
            'AccelerationY(EntityCoord) (m/s2)', 
            'AccelerationZ(EntityCoord) (m/s2)', 
            'RotationX (deg)', 
            'RotationY (deg)', 
            'RotationZ (deg)', 
            'Throttle [0..1]', 
            'Brake [0..1]', 
            'FrontWheelAngle (deg)', 
            'Exceeding Speed (km/h)', 
            'Deficit Speed (km/h)', 
            'TimeToCollision (sec)', 
            'VtV Distance (m)', 
            'WayOff Distance (m)']
    ----------------------------------------
    This function extract vehicle trajectory of each entitiy and show 3D graph

    state_log_df:Pandas dataframe
    elev: related to camera view of 3d plot
    azim: related to camera view of 3d plot
    save: if True, you can save your fig in `./output` folder, if False, you can see plot
    fig_name: if you want to change the plot file name, use it
    '''

    position_x = POSITIONS[0] # dataframe column key 'PositionX (m)'
    position_y = POSITIONS[1] # dataframe column key 'PositionY (m)'
    position_z = POSITIONS[2] # dataframe column key 'PositionZ (m)'

    entitiy_data_dict = {
        entity: state_log_df[state_log_df['Entity'] == entity].reset_index(drop=True)
        for entity in pd.unique(state_log_df['Entity'])
    }

    base_x = entitiy_data_dict["Ego"][position_x].iloc[0]
    base_y = entitiy_data_dict["Ego"][position_y].iloc[0]
    base_z = entitiy_data_dict["Ego"][position_z].iloc[0]

    colors = plt.cm.tab10(np.linspace(0, 1, len(entitiy_data_dict)))
    green = mcolors.to_rgb("green")
    blue = mcolors.to_rgb("blue")
    red = mcolors.to_rgb("red")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    total_velocity = sqrt_square_sum(state_log_df[VELOCITY['x']],
                                     state_log_df[VELOCITY['y']],
                                     state_log_df[VELOCITY['z']],)

    max_velocity = max(total_velocity)

    all_x, all_y, all_z = [], [], []
    for idx, (key, position_data) in enumerate(entitiy_data_dict.items()):

        # 상대 좌표로 이동
        x = position_data[position_x] - base_x
        y = position_data[position_y] - base_y
        z = position_data[position_z] - base_z

        vel_x = position_data[VELOCITY['x']]
        vel_y = position_data[VELOCITY['y']]
        vel_z = position_data[VELOCITY['z']]

        velocity = sqrt_square_sum(vel_x,vel_y,vel_z)

        norm = Normalize(vmin=0, vmax=total_velocity.max())  # or just use velocity.min()/max()
        lc = create_colored_line_segments(x, y, z, velocity, cmap='plasma', norm=norm)
        ax.add_collection3d(lc)

        # Line color -> plt.cm.tab10 based
        line_color = colors[idx]

        if key=='Ego':
            current_color = green
        else: 
            current_color = blue

        # 시작점 (첫 좌표) && 끝점 (마지막 좌표)
        start_color = blend_colors(current_color, line_color,alpha=0.2)
        end_color = blend_colors(red, line_color, alpha=0.2)

        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=start_color, marker='o', s=50, label=f"{key} start")
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=end_color, marker='x', s=50, label=f"{key} end")

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

    ax.set_xlabel(f"{position_x.split()[0]} (relative, regulated (m))")
    ax.set_ylabel(f"{position_y.split()[0]} (relative, regulated (m))")
    ax.set_zlabel(f"{position_z.split()[0]} (relative, regulated (m))")
    ax.set_title("3D Line Plot (Origin = Ego Vehicle First Point)")

    ax.view_init(elev=elev, azim=azim)

    all_data = np.array([
        pd.concat(all_x).to_numpy(),
        pd.concat(all_y).to_numpy(),
        pd.concat(all_z).to_numpy()
    ])

    min_vals = all_data.min(axis=1)
    max_vals = all_data.max(axis=1)
    centers = (min_vals + max_vals) / 2
    max_range = (max_vals - min_vals).max() / 2

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

    # 범례 표시
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)

    ticks = np.linspace(0, max_velocity, num=5)
    cbar = plt.colorbar(lc, ax=ax, pad=0.2, shrink=0.7)
    cbar.set_label("Velocity (m/s)")
    cbar.ax.text(1.05, 1.02, f"Max: {max_velocity:.2f}", transform=cbar.ax.transAxes,
             ha='left', va='bottom', fontsize=9, color='black')

    plt.tight_layout()

    if save:
        output_dir = "./output2"
        os.makedirs(output_dir, exist_ok=True)

        # 파일명 없으면 기본값 사용
        fig_name = fig_name or "plot.png"

        # 확장자 보장
        if not fig_name.endswith('.png'):
            fig_name += '.png'

        # 저장
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300, bbox_inches='tight')
        print(f"✅ 그림 저장됨: {os.path.join(output_dir, fig_name)}")
    else:
        plt.show()

    plt.close(fig)

def main(save=False):

    all_log_folder = [log_folder for log_folder in os.listdir(LOG_FOLDER_PATH) if Path(log_folder).is_dir]
    
    for scenario_log_folder in all_log_folder:
        # ../logs_scenario_runner 안에 시나리오 실행 이후 1개의 로그 폴더 생성
        # 생성된 모든 시뮬레이트 로그 폴더에 대해서 실행
         
        scenario_folder_path = LOG_FOLDER_PATH / scenario_log_folder
        print(scenario_folder_path)
        scenario_log_files = sep_log_file(scenario_folder_path)
        state_log = scenario_log_files['statelog']
        event_log = scenario_log_files['eventlog']
        result_log = scenario_log_files['result']

        sceanrio_file_name = None
        if result_log:
            try:
                result_data = pd.read_csv(scenario_folder_path / result_log)
                sceanrio_file_name = result_data["File Name"][0]
            except EmptyDataError:
                print(f"❌ {result_log}: result_log - 빈 파일 (헤더도 없음) → 건너뜀")
                continue
        else:
            sceanrio_file_name = scenario_log_folder

        try:
            state_data = pd.read_csv(scenario_folder_path / state_log)
            if state_data.empty:
                continue
        except EmptyDataError:
            print(f"❌ {sceanrio_file_name}: statelog - 빈 파일 (헤더도 없음) → 건너뜀")
            continue

        plot_3d_line_equal_unit(state_data, elev=30, azim=70, save=save, fig_name=sceanrio_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='저장 여부, 저장시 ')

    args = parser.parse_args()
    main(save=args.save)
    # main()

# %%
