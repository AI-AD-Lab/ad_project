import pandas as pd
from _utils.plot_utils import *
from rule_utils.common_util import rolling

if __name__=='__main__':

    example_statelog_csv = r"..\simulation_TOTAL_250626\20250617_152628_R_KR_PG_KATRI_LRST1_01_statelog.csv"
    example_df = pd.read_csv(example_statelog_csv)

    plot_2d_graph(example_df, x='time (sec)', y='VelocityY(EntityCoord) (km/h)', save_path=None)
    rolling_data = rolling(example_df)
    plot_2d_graph(rolling_data, x='time (sec)', y='VelocityY(EntityCoord) (km/h)', save_path=None)

    plot_maker = driving_trajectory_plotter(entity="Ego", color_map='viridis')
    plot_maker.plot_2d(example_df, visualize_velocity=True, save_path=None)
    plot_maker.plot_3d(example_df, visualize_velocity=True, save_path=None, elev=30, azim=-110)

