# IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from config import config
import itertools
from multiprocessing import Pool, cpu_count


from _utils.data_processing_utils import data_load
from _utils.plot_utils import plot_confusion_matrix_table

from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

'''
Folder Structure:
    |- AD_Project (Project Root)
        |- greedy_search_priority.py  <--- This file
    |- simulator_stateliog_data_dir (simulation_TOTAL_250626)
        |- 20250617_152628_R_KR_PG_KATRI_LRST1_01_statelog.csv
        |- label.csv
        |- ...
'''

#CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
# SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
# labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']
# labels = ['ST','RA', 'UT', 'LT', 'RT', 'LLC', 'RLC']
# labels = ['UT', 'RA', 'LT', 'RT', 'ST', 'LLC', 'RLC']
labels = ['RA', 'UT', 'LT', 'RT', 'ST', 'LLC', 'RLC']
perms = list(itertools.permutations(labels))

# MAIN CLASSIFICATION FUNCTION
def excute_rule_based_classification(class_perm:list[str]) -> pd.DataFrame:

    labeled_data = [[] for _ in range(len(labels))] # empty list for each class
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SYN_LOG_DATA_ROOT_DIR / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        COUNT = 1

        LLC = detect_left_lane_change(data, duration_sec=0.875357, threshold=0.212362)
        RLC = detect_right_lane_change(data, duration_sec=0.833088, threshold=0.137831)
        ST = detect_straight(data, abs_normal_threshold=0.059045, abs_threshold=0.109141, duration_sec=11.658522)
        RT = detect_right_turn(data, right_threshold=1.183795, duration_sec=1.574704)
        LT = detect_left_turn(data, left_threshold=-0.427932, duration_sec=4.137019)
        RA = detect_roundabout(data, threshold_neg=-0.146373, threshold_pos=0.097355, duration_sec=1.454986, max_duration_sec=14.449369)
        UT = detect_u_turn(data, threshold=-2.187467, duration_sec=2.823866)

        label_variable = {
            'RA': RA, 'ST': ST, 'UT': UT,
            'LT': LT, 'RT': RT, 'LLC': LLC, 'RLC': RLC
        }

        values = [label_variable[label] for label in class_perm]
        for i, value in enumerate(values):
            if value:
                result_list = [0] * 9
                result_list[i] = 1
                break
            else:
                result_list = [0] * 9
                result_list[-2] = 1  # NO_LABEL

        result_list[-1] = COUNT
        labeled_data[real_index[label]].append(result_list)

    total_result = []
    for i, label in enumerate(labeled_data):
        np_sliced = np.array(label)
        np_sliced_change = [[0 if x is None else x for x in row] for row in np_sliced]
        column_sum = np.sum(np_sliced_change, axis=0)
        total_result.append(column_sum)

    perm_array = list(class_perm)
    df_total_result = pd.DataFrame(total_result, columns=perm_array + ["NO_LABEL", "TOTAL"], index=perm_array)

    return df_total_result


def optimiezd_classification(class_perm:None|list) -> pd.DataFrame:
    # 준비: 행 인덱스(정답 라벨 → 행번호), 열 정의
    
    if class_perm is None:
        class_perm = labels
    else:
        class_perm = class_perm
    
    perm_array = list(class_perm)
    cols = perm_array + ["NO_LABEL", "TOTAL"]
    k = len(perm_array)

    # 각 정답 라벨(행)에 대한 누적합 벡터 초기화
    # totals[row_idx] = [0]*(k+2)
    totals = {}
    for idx, lab in enumerate(perm_array):
        totals[idx] = [0] * (k + 2)

    # 정답 라벨을 행 인덱스로 매핑 (원코드 호환)
    real_index = {short_to_long_label[label]: idx for idx, label in enumerate(class_perm)}

    # detect 함수 디스패처 (순서대로 평가 → True면 즉시 중단)
    def eval_in_order(data):
        # class_perm 순서대로 필요할 때만 평가
        for i, lab in enumerate(perm_array):
            if lab == 'LLC':
                if detect_left_lane_change(data, duration_sec=0.875357, threshold=0.212362):
                    return i
            elif lab == 'RLC':
                if detect_right_lane_change(data, duration_sec=0.833088, threshold=0.137831):
                    return i
            elif lab == 'ST':
                if detect_straight(data, abs_normal_threshold=0.059045, abs_threshold=0.109141, duration_sec=7.532441):
                    return i
            elif lab == 'RT':
                if detect_right_turn(data, right_threshold=1.183795, duration_sec=1.574704, max_duration_sec=5.47996):
                    return i
            elif lab == 'LT':
                if detect_left_turn(data, left_threshold=-0.427932, duration_sec=4.137019, max_duration_sec=6.156220):
                    return i
            elif lab == 'RA':
                if detect_roundabout(data, threshold_neg=-0.146373, threshold_pos=0.097355,
                                     duration_sec=1.454986, max_duration_sec=14): #14.449369
                    return i
            elif lab == 'UT':
                if detect_u_turn(data, threshold=2.187467, duration_sec=2.823866):
                    return i
        return None  # 아무 클래스에도 해당 안 됨

    # 메인 루프: 원패스 처리(집계 즉시 반영)
    for file, gt_label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SYN_LOG_DATA_ROOT_DIR / file
        data = data_load(file_path)  # I/O + 파싱 비용

        pred_idx = eval_in_order(data)  # 조기 종료 가능

        row_idx = real_index[gt_label]  # 정답 라벨이 가리키는 행
        row = totals[row_idx]

        if pred_idx is not None:
            row[pred_idx] += 1
        else:
            row[k] += 1  # NO_LABEL
        row[k + 1] += 1  # TOTAL (COUNT=1)

    # DataFrame 생성
    total_result = [totals[i] for i in range(len(perm_array))]
    df_total_result = pd.DataFrame(total_result, columns=cols, index=perm_array)
    return df_total_result

# MULTIPROCESSING WRAPPER
def process_one_perm(args):
    perm_idx, perm = args
    df_total_result = optimiezd_classification(class_perm=perm)

    save_dir = Path('./output/score_data_st')
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_dir / f"total_result_{perm_idx}.png"
    csv_path = save_dir / f"total_result_{perm_idx}.csv"

    plot_confusion_matrix_table(df_total_result, save_path=str(plot_path))
    df_total_result.to_csv(csv_path, index=False)

    if perm_idx % 20 == 0:
        print(f"[INFO] Processed {perm_idx} permutations.")

#%% MAIN RUN
if __name__ == "__main__":
    num_workers = min(cpu_count(), 8)
    print(f"Starting multiprocessing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        pool.map(process_one_perm, list(enumerate(perms)))

