from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import seaborn as sns



plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# Simple DataFrame 2d plot
def plot_2d_graph(df, x='time (sec)', y='speed',
                  legend:bool=True,
                  save_path:None|str=None):

    y_low_lim = min(df[y].min() - abs(df[y].min())*0.1 , -0.5)
    y_up_lim = max(df[y].max() + abs(df[y].max())*0.1, 0.5)

    plt.figure(figsize=(6, 5))
    plt.plot(df[x], df[y], color='#1f77b4', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=2)
    plt.xlim(df[x].min(), df[x].max())
    plt.ylim(y_low_lim, y_up_lim)

    plt.xlabel(f"{x}", fontsize=14, fontweight='bold')
    plt.ylabel(f"{y}", fontsize=14, fontweight='bold')

    if legend:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return

    plt.show()
    plt.close()

def plot_wheel_and_position(data:pd.DataFrame,
            save_path:str|None=None, ):

    base_axis = 'time (sec)'
    first_axis = 'FrontWheelAngle (deg)'
    second_axis = 'PositionX (m)'
    third_axis = 'PositionY (m)'

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 첫번째 그림: Front Wheel Angle
    # 범위를 -1~ 1로 제한
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 0].plot(data[base_axis], data[first_axis], label=first_axis, color='blue')
    axs[0, 0].set_title(f'{first_axis} Over {base_axis}')
    axs[0, 0].set_xlabel(base_axis)
    axs[0, 0].set_ylabel(f'{first_axis}')
    axs[0, 0].grid()
    axs[0, 0].legend()

    # 두번째 그림: 2D plot of x and y coordinates
    axs[0, 1].plot(data[second_axis], data[third_axis], label='Trajectory', color='green')
    axs[0, 1].set_title('2D Trajectory Plot')
    axs[0, 1].set_xlabel(second_axis)
    axs[0, 1].set_ylabel(third_axis)
    axs[0, 1].grid()
    axs[0, 1].legend()

    # 세번째 그림: Time vs PositionX
    axs[1, 0].plot(data[base_axis], data[second_axis], label=second_axis, color='red')
    axs[1, 0].set_title(f'Time vs {second_axis}')
    axs[1, 0].set_xlabel(base_axis)
    axs[1, 0].set_ylabel(second_axis)
    axs[1, 0].grid()
    axs[1, 0].legend()

    # 네번째 그림: Time vs PositionY
    axs[1, 1].plot(data[base_axis], data['PositionY (m)'], label='PositionY', color='purple')
    axs[1, 1].set_title('Time vs PositionY')
    axs[1, 1].set_xlabel(base_axis)
    axs[1, 1].set_ylabel('PositionY (m)')
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.tight_layout()
    if save_path:
        if not save_path.endswith('.png'):
            save_path += '.png'
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_confusion_matrix_table(df, save_path:None|str=None):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.scale(1, 1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

class driving_trajectory_plotter:

    def __init__(self, entity:str="Ego",
                 color_map:str='viridis'):

        self.entity = entity
        self.color_map = color_map

        self.position_x = 'PositionX (m)'
        self.position_y = 'PositionY (m)'
        self.position_z = 'PositionZ (m)'

        self.green = mcolors.to_rgb("green")
        self.red = mcolors.to_rgb("red")
        self.blue = mcolors.to_rgb("blue")

        self.VELOCITY = {
                'x': 'VelocityX(EntityCoord) (km/h)',
                'y': 'VelocityY(EntityCoord) (km/h)',
                'z': 'VelocityZ(EntityCoord) (km/h)'}

    def _sqrt_square_sum(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        return np.sqrt(x**2 + y**2 + z**2)

    def _create_colored_line_segments(self, x, y, z, velocity, norm=None):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        x, y, z, velocity = map(np.asarray, (x, y, z, velocity))
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if norm is None:
            norm = Normalize(vmin=velocity.min(), vmax=velocity.max())

        lc = Line3DCollection(segments, cmap=self.color_map, norm=norm)
        lc.set_array(velocity[:-1])
        lc.set_linewidth(2)

        return lc

    def plot_3d(self,
                state_log_dataframe:pd.DataFrame,
                visualize_velocity:bool=False,
                save_path:None|str|Path=None,
                elev:int=30,
                azim:int=-110):

        entity_state_log_data = state_log_dataframe[state_log_dataframe['Entity'] == self.entity]
        base_x = entity_state_log_data[self.position_x].iloc[0]
        base_y = entity_state_log_data[self.position_y].iloc[0]
        base_z = entity_state_log_data[self.position_z].iloc[0]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(f"{self.position_x.split()[0]}")
        ax.set_ylabel(f"{self.position_y.split()[0]}")
        ax.set_zlabel(f"{self.position_z.split()[0]}")

        x = entity_state_log_data[self.position_x] - base_x
        y = entity_state_log_data[self.position_y] - base_y
        z = entity_state_log_data[self.position_z] - base_z

        ax.plot3D(x, y, z, linewidth=2, color=self.green)
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color=self.red, marker='o', s=50, label=f"start point")
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color=self.blue, marker='x', s=50, label=f"end point")

        ax.view_init(elev=elev, azim=azim)

        min_vals = np.array([x,y,z]).min(axis=1)
        max_vals = np.array([x,y,z]).max(axis=1)
        centers = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2

        ax.set_xlim(centers[0] - max_range*1.1, centers[0] + max_range*1.1)
        ax.set_ylim(centers[1] - max_range*1.1, centers[1] + max_range*1.1)
        ax.set_zlim(centers[2] - max_range*1.1, centers[2] + max_range*1.1)

        if visualize_velocity:
            velocity_x = entity_state_log_data[self.VELOCITY['x']]
            velocity_y = entity_state_log_data[self.VELOCITY['y']]
            velocity_z = entity_state_log_data[self.VELOCITY['z']]

            entity_velocity = self._sqrt_square_sum(velocity_x, velocity_y, velocity_z)
            max_velocity = max(entity_velocity)

            norm = Normalize(vmin=0, vmax=max_velocity)
            lc = self._create_colored_line_segments(x, y, z, entity_velocity, norm=norm)
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

    def plot_2d(self,
                state_log_dataframe:pd.DataFrame,
                visualize_velocity:bool=False,
                save_path:None|str|Path=None):

        entity_state_log_data = state_log_dataframe[state_log_dataframe['Entity'] == self.entity]
        base_x = entity_state_log_data[self.position_x].iloc[0]
        base_y = entity_state_log_data[self.position_y].iloc[0]

        fig_trajectory = plt.figure(figsize=(6, 6), dpi=300)
        ax = fig_trajectory.add_subplot(111)
        ax.set_xlabel(f"{self.position_x.split()[0]}")
        ax.set_ylabel(f"{self.position_y.split()[0]}")

        x = entity_state_log_data[self.position_x] - base_x
        y = entity_state_log_data[self.position_y] - base_y

        ax.scatter(x.iloc[0], y.iloc[0], color=self.red, marker='o', s=50, label=f"start point")
        ax.scatter(x.iloc[-1], y.iloc[-1], color=self.blue, marker='x', s=50, label=f"end point")

        min_vals = np.array([x,y]).min(axis=1)
        max_vals = np.array([x,y]).max(axis=1)
        centers = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2

        ax.set_xlim(centers[0] - max_range*1.1, centers[0] + max_range*1.1)
        ax.set_ylim(centers[1] - max_range*1.1, centers[1] + max_range*1.1)

        if visualize_velocity:
            from matplotlib.collections import LineCollection

            velocity_x = entity_state_log_data[self.VELOCITY['x']]
            velocity_y = entity_state_log_data[self.VELOCITY['y']]
            velocity_z = entity_state_log_data[self.VELOCITY['z']]

            entity_velocity = self._sqrt_square_sum(velocity_x, velocity_y, velocity_z)
            max_velocity = max(entity_velocity)

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = Normalize(vmin=0, vmax=max_velocity)
            lc = LineCollection(segments, cmap=self.color_map, norm=norm)
            lc.set_array(entity_velocity[:-1])
            lc.set_linewidth(5)
            ax.add_collection(lc)

            handles, labels = ax.get_legend_handles_labels()

            if save_path:
                # 🎯 save main trajectory
                fig_trajectory.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig_trajectory)

                # 🎯 save legend
                fig_legend, ax_legend = plt.subplots(figsize=(2.5, 1.0))  # 사이즈 조정 가능
                ax_legend.axis('off')  # 축 제거
                ax_legend.legend(handles, labels, loc='center', fontsize=10, frameon=False)

                legend_savepath = save_path.replace('.png', '_legend.png')
                fig_legend.savefig(legend_savepath, dpi=300, bbox_inches='tight', transparent=True)
                plt.close(fig_legend)

                # 🎯 save colorbar
                fig_colorbar, ax_colorbar = plt.subplots(figsize=(1.0, 4.0))
                fig_colorbar.subplots_adjust(left=0.5, right=0.8)

                sm = ScalarMappable(norm=norm, cmap=self.color_map)
                sm.set_array([])

                cbar = fig_colorbar.colorbar(sm, cax=ax_colorbar)
                cbar.set_label("Velocity (km/h)")

                colorbar_savepath = save_path.replace('.png', '_colorbar.png')
                fig_colorbar.savefig(colorbar_savepath, dpi=300, bbox_inches='tight')
                plt.close(fig_colorbar)
            else:
                plt.show()

        else:
            ax.scatter(x, y, color=self.green)
            if save_path:
                fig_trajectory.savefig(save_path, dpi=300)
                plt.close(fig_trajectory)
            else:
                plt.show()


def plot_pretty_broken_hist(y):
    y = np.array(y) / 50
    bin_width = 2
    bins = np.arange(0, y.max() + bin_width, bin_width)
    counts, edges = np.histogram(y, bins=bins)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(8, 3.8),
        gridspec_kw={'wspace': 0.06, 'width_ratios': [3, 2]}
    )

    # left and right axes common settings
    for ax in (ax1, ax2):
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)

    # histogram bars
    color = '#1f77b4'
    ax1.bar(edges[:-1], counts, width=bin_width*0.85,
            color=color, alpha=0.8, edgecolor='black', linewidth=0.6)
    ax2.bar(edges[:-1], counts, width=bin_width*0.85,
            color=color, alpha=0.8, edgecolor='black', linewidth=0.6)

    ax1.set_xlim(0, 55)
    ax2.set_xlim(145, 185)

    # elimenate unnecessary spines and ticks
    ax1.spines['right'].set_visible(False)  
    ax2.spines['left'].set_visible(False)   
    ax2.tick_params(labelleft=False)     

    # break mark settings
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    cx = (pos1.x1 + pos2.x0)/2.0 + 0.002             
    bottom_y = pos1.y0

    gap_x = pos1.x1
    gap_w = (pos2.x0 - pos1.x1)
    offset = 0.002

    # eliminate unnecessary spines and ticks
    fig.add_artist(Rectangle((gap_x + offset, 0.2), gap_w, 0.65,
                            transform=fig.transFigure,
                            facecolor='white', edgecolor='none',
                            zorder=5))
    # add '~' symbol between axes to indicate break at the bottom
    fig.text(cx, bottom_y, '≈', rotation=90, ha='center', va='center',
             fontsize=14, weight='bold', clip_on=False)

    fig.text(0.5, -0.02, 'Duration (s)', ha='center', fontsize=20)
    ax1.set_ylabel('Number of Samples', fontsize=20, fontweight='bold')

    axis_font_size = 14
    ax1.tick_params(axis='y', labelsize=axis_font_size)
    ax1.tick_params(axis='x', labelsize=axis_font_size)
    ax2.tick_params(axis='x', labelsize=axis_font_size)

    for ax in (ax1, ax2):
        ax.grid(False)  # 전체 격자 제거
        ax.grid(True, axis='y', linestyle='--', alpha=0.35, clip_on=False)

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 1.0])
    # plt.subplots_adjust(bottom=0.18, top=0.95, left=0.08, right=0.98, wspace=0.08)
    plt.show()


def plot_correlation_heatmap(df, method='pearson', figsize=(8, 6), cmap='coolwarm', title=None):

    # select numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        raise ValueError("No numeric columns found in the DataFrame.")

    # compute correlation matrix
    corr_df = df_numeric.corr(method=method)

    #  plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        annot=True,        
        fmt=".2f",      
        cmap=cmap,
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8}
    )


    plt.title(title or f"Correlation Matrix ({method.title()})", fontsize=14, fontweight='bold', fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman', fontsize=10, rotation=45)
    plt.yticks(fontname='Times New Roman', fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    return corr_df





if __name__=='__main__':

    example_statelog_csv = r"D:\3세부 Tier4\scenario_ws\simulation_TOTAL_250626\20250617_152628_R_KR_PG_KATRI_LRST1_01_statelog.csv"
    example_df = pd.read_csv(example_statelog_csv)

    plot_2d_graph(example_df, x='time (sec)', y='VelocityY(EntityCoord) (km/h)', save_path=None)

    # plot_maker = driving_trajectory_plotter(entity="Ego", color_map='viridis')
    # plot_maker.plot_2d(example_df, visualize_velocity=True, save_path=None)
    # plot_maker.plot_3d(example_df, visualize_velocity=True, save_path=None, elev=30, azim=-110)



# %%
