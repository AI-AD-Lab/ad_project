from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import colors as mcolors
import argparse
from pandas.errors import EmptyDataError

# main.py 기준으로 상위 디렉터리 && target_folder 내부 파일 - target_folder: '../logs_scenario_runner'
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"
POSITIONS = ['PositionX (m)', 'PositionY (m)', 'PositionZ (m)']

def sep_log_file(log_folder_path):
    
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

def plot_3d_line_equal_unit(state_log_df, elev=30, azim=70, save=False, fig_name:str|None=None):


    '''
    state_log_df columns: 
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
    This function extract vehicle trajectory of each entity and show 3D graph


    state_log_df:Pandas dataframe
    elev: related to camera view of 3d plot
    azim: related to camera view of 3d plot
    save: if True, you can save your fig in `./output` folder, if False, you can see plot
    fig_name: if you want to change the plot file name, use it
    '''


    position_x = POSITIONS[0] # dataframe column key 'PositionX (m)'
    position_y = POSITIONS[1] # dataframe column key 'PositionY (m)'
    position_z = POSITIONS[2] # dataframe column key 'PositionZ (m)'

    entity_data_dict = {
        entity: state_log_df[state_log_df['Entity'] == entity].reset_index(drop=True)
        for entity in pd.unique(state_log_df['Entity'])
    }

    base_x = entity_data_dict["Ego"][position_x].iloc[0]
    base_y = entity_data_dict["Ego"][position_y].iloc[0]
    base_z = entity_data_dict["Ego"][position_z].iloc[0]

    colors = plt.cm.tab10(np.linspace(0, 1, len(entity_data_dict)))
    green = mcolors.to_rgb("green")
    red = mcolors.to_rgb("red")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    all_x, all_y, all_z = [], [], []
    for idx, (key, position_data) in enumerate(entity_data_dict.items()):

        # 상대 좌표로 이동
        x = position_data[position_x] - base_x
        y = position_data[position_y] - base_y
        z = position_data[position_z] - base_z

        # Line color -> plt.cm.tab10 based
        line_color = colors[idx]
        ax.plot3D(x, y, z, linewidth=2, color=line_color, label=key)

        # 시작점 (첫 좌표) && 끝점 (마지막 좌표)
        start_color = blend_colors(green, line_color,alpha=0.2)
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
    ax.legend()
    plt.tight_layout()

    if save:
        output_dir = "./output"
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

        scenario_file_name = None
        if result_log:
            try:
                result_data = pd.read_csv(scenario_folder_path / result_log)
                scenario_file_name = result_data["File Name"][0]
            except EmptyDataError:
                print(f"❌ {result_log}: result_log - 빈 파일 (헤더도 없음) → 건너뜀")
                continue
        else:
            scenario_file_name = scenario_log_folder

        try:
            state_data = pd.read_csv(scenario_folder_path / state_log)
            if state_data.empty:
                continue
        except EmptyDataError:
            print(f"❌ {scenario_file_name}: statelog - 빈 파일 (헤더도 없음) → 건너뜀")
            continue

        plot_3d_line_equal_unit(state_data, elev=30, azim=70, save=save, fig_name=scenario_file_name)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='저장 여부, 저장시 ')

    args = parser.parse_args()
    main(save=args.save)

# %%
