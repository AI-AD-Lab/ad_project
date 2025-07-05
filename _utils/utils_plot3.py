from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
# plt.style.use(['science','ieee'])

POSITIONS = {'x':'PositionX (m)',
             'y':'PositionY (m)',
             'z':'PositionZ (m)'}

VELOCITY = {'x': 'VelocityX(EntityCoord) (km/h)',
            'y': 'VelocityY(EntityCoord) (km/h)',
            'z': 'VelocityZ(EntityCoord) (km/h)'}

color_map = ["viridis", "plasma", "magma", "inferno", "cividis"]

def sqrt_square_sum( x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    return np.sqrt(x**2 + y**2 + z**2)

def plot_driving_trajectory(state_log_dataframe:pd.DataFrame,
                            entity:str="Ego",
                            visualize_velocity:bool=False,
                            color_map:str='viridis',
                            save_path:None|str|Path=None,
                            ):

    position_x = POSITIONS['x']
    position_y = POSITIONS['y']

    entity_state_log_data = state_log_dataframe[state_log_dataframe['Entity'] == entity]
    base_x = entity_state_log_data[position_x].iloc[0]
    base_y = entity_state_log_data[position_y].iloc[0]

    green = mcolors.to_rgb("green")
    red = mcolors.to_rgb("red")
    blue = mcolors.to_rgb("blue")

    fig_trajectory = plt.figure(figsize=(6, 6), dpi=300)
    ax_trajectory = fig_trajectory.add_subplot(111)
    ax_trajectory.set_xlabel(f"Position X (m)", fontsize=14, fontweight='bold')
    ax_trajectory.set_ylabel(f"Position Y (m)", fontsize=14, fontweight='bold')


    x = entity_state_log_data[position_x] - base_x
    y = entity_state_log_data[position_y] - base_y


    ax_trajectory.scatter(x.iloc[0], y.iloc[0], color=red, marker='o', s=90, label=f"start point")
    ax_trajectory.scatter(x.iloc[-1], y.iloc[-1], color=blue, marker='x',
                          s=100, label=f"end point")
    # ax_trajectory.legend(fontsize=12, loc='best')


    min_vals = np.array([x,y]).min(axis=1)
    max_vals = np.array([x,y]).max(axis=1)
    centers = (min_vals + max_vals) / 2
    max_range = (max_vals - min_vals).max() / 2

    free_space = 5
    ax_trajectory.set_xlim(centers[0] - max_range - free_space, centers[0] + max_range + free_space)
    ax_trajectory.set_ylim(centers[1] - max_range - free_space, centers[1] + max_range + free_space)

    if visualize_velocity:
        velocity_x = entity_state_log_data[VELOCITY['x']]
        velocity_y = entity_state_log_data[VELOCITY['y']]
        velocity_z = entity_state_log_data[VELOCITY['z']]

        entity_velocity = sqrt_square_sum(velocity_x, velocity_y, velocity_z)
        max_velocity = max(entity_velocity)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=0, vmax=max_velocity)
        lc = LineCollection(segments, cmap=color_map, norm=norm)
        lc.set_array(entity_velocity[:-1])
        lc.set_linewidth(5)
        ax_trajectory.add_collection(lc)

        handles, labels = ax_trajectory.get_legend_handles_labels()

        # 🎯 메인 그래프 저장 (trajectory만)
        if save_path:
            fig_trajectory.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig_trajectory)
        else:
            plt.show()


        fig_legend, ax_legend = plt.subplots(figsize=(2.5, 1.0))  # 사이즈 조정 가능
        ax_legend.axis('off')  # 축 제거
        legend = ax_legend.legend(handles, labels, loc='center', fontsize=10, frameon=False)

        legend_savepath = save_path.replace('.png', '_legend.png')
        fig_legend.savefig(legend_savepath, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_legend)

        # 🎯 Colorbar만 별도 저장
        fig_colorbar, ax_colorbar = plt.subplots(figsize=(1.0, 4.0))
        fig_colorbar.subplots_adjust(left=0.5, right=0.8)

        sm = ScalarMappable(norm=norm, cmap=color_map)
        sm.set_array([])  # 빈 배열로 설정 (값은 norm에서 처리됨)

        cbar = fig_colorbar.colorbar(sm, cax=ax_colorbar)
        cbar.set_label("Velocity (km/h)")
        # cbar.ax.text(1.1, 1.02, f"Max: {max_velocity:.2f}",
        #             transform=cbar.ax.transAxes,
        #             ha='center', va='bottom', fontsize=10, color='black')

        colorbar_savepath = save_path.replace('.png', '_colorbar.png')
        fig_colorbar.savefig(colorbar_savepath, dpi=300, bbox_inches='tight')
        plt.close(fig_colorbar)

    else:
        ax_trajectory.scatter(x, y, color=green)
        if save_path:
            fig_trajectory.savefig(save_path, dpi=300)
            plt.close(fig_trajectory)
        else:
            plt.show()


if __name__=='__main__':
    import os
    example_dir_path = "C:/Users/tndah/Desktop/STAGE_WS/representive_logs/"
    statelog_csv = [statelog for statelog in os.listdir(example_dir_path) if statelog.endswith('statelog.csv')]

    for statelog in statelog_csv:

        example_file_path = example_dir_path + statelog
        example_df = pd.read_csv(example_file_path)

        name_change = statelog.replace('statelog.csv', 'plpt.png')

        saving_path = example_dir_path + name_change
        plot_driving_trajectory(example_df, save_path=saving_path , visualize_velocity=True)
