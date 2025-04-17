#%%
import os
from _utils.utils_path import *
from _utils.utils_plot import *
# %%
MORAISIM_PATH = Path(__file__).resolve().parent.parent
LOG_FOLDER_PATH = MORAISIM_PATH / "logs_scenario_runner"
if __name__ == '__main__':

    test_path = LOG_FOLDER_PATH /'simulation_20250417_183629'
    dd = sep_log_file(test_path)
    for key in dd:
        log_set = dd[key]
        state_log_data = load_data(test_path,log_set)
        # pp = PLOTING(state_log_data)
        # pp.plot_trajectory(show_velocity=True, fig_name=key, save=True)

        
    
# %%
