#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from config import config

from _utils.data_processing_utils import data_load
from _utils.plot_utils import plot_confusion_matrix_table

from rule_utils.straight import SSST_TEST, detect_straight


# CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / "ST_logs"
short_to_long_label = config['Short_to_Long_Label']
ssst_files = os.listdir(SYN_LOG_DATA_ROOT_DIR)
ssst_statelog_files = [file for file in ssst_files if file.endswith("statelog.csv")]

result = []
for file in ssst_statelog_files:
    data_path = SYN_LOG_DATA_ROOT_DIR / file
    # print(data_path)
    data = data_load(data_path)

    sst_cls = detect_straight(data, abs_normal_threshold=0.059045, abs_threshold=0.109141, duration_sec=7.532441)
    if sst_cls != 1:
        print(file)
    # result.append([file, sst_cls])


# %%
