from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import MultipleLocator

'''
state_log_df coloumns:
        ['time (sec)',
        'Entity',                               * REQUIRED
        'PositionX (m)',                        * REQUIRED
        'PositionY (m)',                        * REQUIRED
        'PositionZ (m)',                        * REQUIRED
        'VelocityX(EntityCoord) (km/h)',        * REQUIRED
        'VelocityY(EntityCoord) (km/h)',        * REQUIRED
        'VelocityZ(EntityCoord) (km/h)',]       * REQUIRED
----------------------------------------
This function extract vehicle trajectory of each entitiy and show 3D graph


state_log_df:Pandas dataframe
elev: related to camera view of 3d plot
azim: related to camera view of 3d plot
save: if True, you can save your fig in `./output` folder, if False, you can see plot
fig_name: if you want to change the plot file name, use it
'''


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

def plot_driving_trajectory(state_log_dataframe:pd.DataFrame,
                            entity:str="Ego",
                            visulaize_velocity:bool=False,
                            color_map:str='viridis',
                            save_path:None|str|Path=None,
                            elev=30, azim=-110,
                            ):

    position_x = POSITIONS['x']
    position_y = POSITIONS['y']
    position_z = POSITIONS['z']

    entity_state_log_data = state_log_dataframe[state_log_dataframe['Entity'] == entity]
    base_x = entity_state_log_data[position_x].iloc[0]
    base_y = entity_state_log_data[position_y].iloc[0]
    base_z = entity_state_log_data[position_z].iloc[0]

    green = mcolors.to_rgb("green")
    red = mcolors.to_rgb("red")
    blue = mcolors.to_rgb("blue")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(f"{position_x.split()[0]}")
    ax.set_ylabel(f"{position_y.split()[0]}")
    ax.set_zlabel(f"{position_z.split()[0]}")

    x = entity_state_log_data[position_x] - base_x
    y = entity_state_log_data[position_y] - base_y
    z = entity_state_log_data[position_z] - base_z

    ax.plot3D(x, y, z, linewidth=2, color=green)
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=red, marker='o', s=50, label=f"start point")
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=blue, marker='x', s=50, label=f"end point")

    ax.view_init(elev=elev, azim=azim)

    min_vals = np.array([x,y,z]).min(axis=1)
    max_vals = np.array([x,y,z]).max(axis=1)
    centers = (min_vals + max_vals) / 2
    max_range = (max_vals - min_vals).max() / 2

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

    if visulaize_velocity:
        velocity_x = entity_state_log_data['VelocityX(EntityCoord) (km/h)']
        velocity_y = entity_state_log_data['VelocityY(EntityCoord) (km/h)']
        velocity_z = entity_state_log_data['VelocityZ(EntityCoord) (km/h)']

        entity_velocity = sqrt_square_sum(velocity_x, velocity_y, velocity_z)
        max_velocity = max(entity_velocity)

        norm = Normalize(vmin=0, vmax=max_velocity)
        lc = create_colored_line_segments(x, y, z, entity_velocity, cmap=color_map, norm=norm)
        ax.add_collection3d(lc)

        ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.0), fontsize=9)
        cbar = plt.colorbar(lc, ax=ax, pad=0.2, shrink=0.7)
        cbar.set_label("Velocity (m/s)")
        cbar.ax.text(1.05, 1.02, f"Max: {max_velocity:.2f}", transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=9, color='black')

    else:
        ax.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__=='__main__':
    example_file_path = r"C:\Users\tndah\Desktop\STAGE_WS\representive_logs\20250617_152628_R_KR_PG_KATRI_LRST1_01_statelog.csv"
    saving_path = r"C:\Users\tndah\Desktop\STAGE_WS\representive_logs\output.png"
    example_df = pd.read_csv(example_file_path)

    plot_driving_trajectory(example_df, save_path=saving_path , visulaize_velocity=True)
