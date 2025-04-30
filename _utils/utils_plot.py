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

# Simple DataFrame 2d plot
def dataframe_2d_plot(df, col1='time (sec)', col2='speed'):
    plt.figure(figsize=(6, 5))
    plt.plot(df[col1], df[col2], marker='o')
    plt.title(f"{col1} vs {col2}")
    plt.xlim(df[col1].min(), df[col1].max())
    plt.ylim(df[col2].min(), df[col2].max())

    plt.xlabel(f"{col1}")
    plt.ylabel(f"{col2}")
    
    plt.autoscale(False)
    plt.show()
    plt.close()


class PLOTING():

    def __init__(self, state_log_df:pd.DataFrame):

        self.position_x = POSITIONS['x']
        self.position_y = POSITIONS['y'] 
        self.position_z = POSITIONS['z'] 

        self.entitiy_data_dict = {
            entity: state_log_df[state_log_df['Entity'] == entity].reset_index(drop=True)
            for entity in pd.unique(state_log_df['Entity'])
        }
        
        print(f"Entities: {self.entitiy_data_dict.keys()}")
        print(pd.unique(state_log_df['Entity']))

        self.base_x = self.entitiy_data_dict["Ego"][self.position_x].iloc[0]
        self.base_y = self.entitiy_data_dict["Ego"][self.position_y].iloc[0]
        self.base_z = self.entitiy_data_dict["Ego"][self.position_z].iloc[0]

        total_velocity = sqrt_square_sum(
                                    state_log_df[VELOCITY['x']],
                                    state_log_df[VELOCITY['y']],
                                    state_log_df[VELOCITY['z']]
                                    )
        self.max_velocity = max(total_velocity)

        # color Group
        self.green = mcolors.to_rgb("green")
        self.red = mcolors.to_rgb("red")
        self.blue = mcolors.to_rgb("blue")
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.entitiy_data_dict)))

        # fig settings
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel(f"{self.position_x.split()[0]} (relative, regulated (m))")
        self.ax.set_ylabel(f"{self.position_y.split()[0]} (relative, regulated (m))")
        self.ax.set_zlabel(f"{self.position_z.split()[0]} (relative, regulated (m))")
        self.ax.set_title("3D Line Plot (Origin = Ego Vehicle First Point)")

    def plot_trajectory(self, show_velocity=False, elev=30, azim=70, save=False, show=False, fig_name: str | None = None):

        all_x, all_y, all_z = [], [], []
        for idx, (key, position_data) in enumerate(self.entitiy_data_dict.items()):
            
            print(key)
            # 상대 좌표로 이동
            x = position_data[self.position_x] - self.base_x
            y = position_data[self.position_y] - self.base_y
            z = position_data[self.position_z] - self.base_z

            # Line color -> plt.cm.tab10 based
            line_color = self.colors[idx]
            self.ax.plot3D(x, y, z, linewidth=2, color=line_color, label=key)

            current_color = self.green if key == 'Ego' else self.blue
            start_color = blend_colors(current_color, line_color, alpha=0.2)
            end_color = blend_colors(self.red, line_color, alpha=0.2)

            if show_velocity:
                vel_x = position_data[VELOCITY['x']]
                vel_y = position_data[VELOCITY['y']]
                vel_z = position_data[VELOCITY['z']]
                velocity = sqrt_square_sum(vel_x,vel_y,vel_z)

                norm = Normalize(vmin=0, vmax=self.max_velocity)  
                self.lc = create_colored_line_segments(x, y, z, velocity, cmap='plasma', norm=norm)
                self.ax.add_collection3d(self.lc)

            self.ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=start_color, marker='o', s=50, label=f"{key} start")
            self.ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=end_color, marker='x', s=50, label=f"{key} end")

            all_x.append(x)
            all_y.append(y)
            all_z.append(z)

        self.all_data = np.array([
        pd.concat(all_x).to_numpy(),
        pd.concat(all_y).to_numpy(),
        pd.concat(all_z).to_numpy()
        ])

        self.ax.view_init(elev=elev, azim=azim)

        min_vals = self.all_data.min(axis=1)
        max_vals = self.all_data.max(axis=1)
        centers = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2

        self.ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        self.ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        self.ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
        
        if show_velocity:
            self.ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
            cbar = plt.colorbar(self.lc, ax=self.ax, pad=0.2, shrink=0.7)
            cbar.set_label("Velocity (m/s)")
            cbar.ax.text(1.05, 1.02, f"Max: {self.max_velocity:.2f}", transform=cbar.ax.transAxes,
                    ha='left', va='bottom', fontsize=9, color='black')
        else:
            self.ax.legend()

        plt.tight_layout()

        if save:
            self.save_plot(fig_name)

        if show:
            plt.show()
        
        plt.close(self.fig)

    def save_plot(self, fig_name: str | None = None):
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        fig_name = (fig_name or "plot") + ".png" if not fig_name or not fig_name.endswith(".png") else fig_name
        save_path = os.path.join(output_dir, fig_name)

        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그림 저장됨: {save_path}")

    def get_trajectory_data(self):
        return self.all_data

    def __del__(self):
        if self.fig:
            plt.close(self.fig)
            print("🧹 Figure closed on object deletion.")

