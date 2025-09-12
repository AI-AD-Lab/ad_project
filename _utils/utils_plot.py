import os

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scienceplots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
def dataframe_2d_plot(df, col1='time (sec)', col2='speed', rolling_window=100, sampling_hz=50):

    # data smoothing, reduce noise
    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02

    plt.figure(figsize=(6, 5))
    plt.plot(df_rolling[col1], df_rolling[col2], marker='o')
    plt.title(f"{col1} vs {col2}")
    plt.xlim(df_rolling[col1].min(), df_rolling[col1].max())
    plt.ylim(df_rolling[col2].min(), df_rolling[col2].max())

    plt.xlabel(f"{col1}")
    plt.ylabel(f"{col2}")

    plt.autoscale(False)
    plt.show()
    plt.close()

def time_base_plot(data:pd.DataFrame,
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


def draw_acceleration_y_plot(df, save_path:None|str|Path = None, rolling_window:int = 100, sampling_hz:int = 50):

    # data smoothing, reduce noise
    df_copy = df.loc[:, ~df.columns.isin(['Entity'])].copy()
    df_rolling = df_copy.rolling(rolling_window).mean().bfill()
    df_rolling['time (sec)'] = df_rolling.index * (1/sampling_hz) # index * 0.02
    ay = df_rolling['AccelerationY(EntityCoord) (m/s2)'].values

    # plt.style.use(['science','ieee'])

    plt.figure(figsize=(10, 4))
    plt.plot(df['time (sec)'], ay, label='Acc Y', color='#1f77b4', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=2)

    if abs(ay).max() < 0.1:
        plt.ylim(-0.5, 0.5)

    plt.xlabel('Time(sec)', fontsize=14, fontweight='bold')
    plt.ylabel('Lateral Acceleration(m/s²)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return

    plt.show()