# pip install optuna
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import random
import optuna

from _utils.data_processing_utils import data_load
from _utils.plot_utils import plot_confusion_matrix_table


from config import config
from rule_utils.lane_change import *
from rule_utils.straight import *
from rule_utils.left_turn import *
from rule_utils.right_turn import *
from rule_utils.roundabout import *
from rule_utils.straight import *
from rule_utils.u_turn import *
from _utils.score_utils import compute_score


#CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']

short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']

# MAIN CLASSIFICATION FUNCTION
def excute_rule_based_classification(parameterset) -> pd.DataFrame:

    LLC_params = parameterset['llc_params']
    RLC_params = parameterset['rlc_params']
    LT_params = parameterset['lt_params']
    RT_params = parameterset['rt_params']
    RA_params = parameterset['ra_params']
    ST_params = parameterset['st_params']
    UT_params = parameterset['ut_params']

    class_perm = labels
    labeled_data = [[] for _ in range(len(labels))] # empty list for each class
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    for file, label in zip(label_data['file_name'], label_data['trajectory type']):
        file_path = SYN_LOG_DATA_ROOT_DIR / file
        data = data_load(file_path)

        ST , RT, LT, UT, LLC, RLC, RA = 0, 0, 0, 0, 0, 0, 0
        COUNT = 1

        LLC = detect_left_lane_change(data, threshold=LLC_params['threshold'], duration_sec=LLC_params['duration_sec'])
        RLC = detect_right_lane_change(data, threshold=RLC_params['threshold'], duration_sec=RLC_params['duration_sec'])
        ST = detect_straight(data, abs_normal_threshold=ST_params['abs_normal_threshold'], abs_threshold=ST_params['abs_threshold'], duration_sec=ST_params['duration_sec'])
        RT = detect_right_turn(data, right_threshold=RT_params['right_threshold'], duration_sec=RT_params['duration_sec'])
        LT = detect_left_turn(data, left_threshold=LT_params['left_threshold'], duration_sec=LT_params['duration_sec'])
        RA = detect_roundabout(data, threshold_neg=RA_params['threshold_neg'], threshold_pos=RA_params['threshold_pos'], duration_sec=RA_params['duration_sec'], max_duration_sec=RA_params['max_duration_sec'] )
        UT = detect_u_turn(data, threshold=UT_params['threshold'], duration_sec=UT_params['duration_sec'])

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


def objective(trial):

    # # 1) 탐색 공간 정의 (예시 범위)
    # llc_threshold   = trial.suggest_float("llc_threshold", 0.02, 0.8)
    # llc_duration    = trial.suggest_float("llc_duration_sec", 0.3, 2.0)

    # rlc_threshold   = trial.suggest_float("rlc_threshold", 0.02, 0.8)
    # rlc_duration    = trial.suggest_float("rlc_duration_sec", 0.3, 2.0)

    # lt_left_th      = trial.suggest_float("lt_left_threshold", -2.5, -0.2)   # 음수(좌회전) 쪽
    # lt_duration     = trial.suggest_float("lt_duration_sec", 1.0, 6.0)

    # rt_right_th     = trial.suggest_float("rt_right_threshold", 0.2, 2.5)    # 양수(우회전) 쪽
    # rt_duration     = trial.suggest_float("rt_duration_sec", 1.0, 6.0)

    # st_abs_norm_th  = trial.suggest_float("st_abs_normal_threshold", 0.01, 0.2)
    # st_abs_th       = trial.suggest_float("st_abs_threshold", 0.15, 0.8)
    # st_duration     = trial.suggest_float("st_duration_sec", 4.0, 12.0)

    # ra_th_neg       = trial.suggest_float("ra_threshold_neg", -1.5, -0.05)   # 음측
    # ra_th_pos       = trial.suggest_float("ra_threshold_pos",  0.05,  1.5)   # 양측
    # ra_duration     = trial.suggest_float("ra_duration_sec", 0.5, 4.0)
    # ra_max_dur      = trial.suggest_float("ra_max_duration_sec", 4.0, 15.0)

    # ut_threshold    = trial.suggest_float("ut_threshold", 30.0, 300.0)       # (예) heading 누적각 등
    # ut_duration     = trial.suggest_float("ut_duration_sec", 2.0, 10.0)


    # V2
    # LLC
    llc_threshold = trial.suggest_float("llc_threshold", 0.1, 0.4)
    llc_duration  = trial.suggest_float("llc_duration_sec", 0.4, 0.9)

    # RLC
    rlc_threshold = 0.137831
    rlc_duration  = 0.833088

    # ST
    st_abs_norm_th = 0.059045
    st_abs_th      = 0.109141
    st_duration    = 11.658522

    # RT
    rt_right_th = 1.183795
    rt_duration = 1.574704

    # LT (typo 주의: left_threshold)
    lt_left_th  = -0.427932
    lt_duration = 4.137019

    # RA
    ra_th_neg    = -0.146373
    ra_th_pos    = 0.097355
    ra_duration  = 1.454986
    ra_max_dur   = 14.449369

    # UT
    ut_threshold = 2.187467
    ut_duration  = 2.823866
    # 2) 질문 코드의 parameterset 포맷에 맞춰 구성
    parameterset = {
        'llc_params':{
            "threshold": llc_threshold,
            "duration_sec": llc_duration
        },
        'rlc_params':{
            "threshold": rlc_threshold,
            "duration_sec": rlc_duration
        },
        'lt_params':{
            "left_threshold": lt_left_th,
            "duration_sec": lt_duration
        },
        'rt_params':{
            "right_threshold": rt_right_th,
            "duration_sec": rt_duration
        },
        'st_params':{
            "abs_normal_threshold": st_abs_norm_th,
            "abs_threshold": st_abs_th,
            "duration_sec": st_duration
        },
        'ra_params':{
            "threshold_neg": ra_th_neg,
            "threshold_pos": ra_th_pos,
            "duration_sec": ra_duration,
            "max_duration_sec": ra_max_dur
        },
        'ut_params':{
            "threshold": ut_threshold,
            "duration_sec": ut_duration
        },
    }

    # 3) 규칙 기반 분류 실행 → 혼동표 DataFrame 산출
    df_total_result = excute_rule_based_classification(parameterset)

    # 4) 점수 계산 (compute_score가 스칼라를 반환한다고 가정)
    score_dict = compute_score(df_total_result)
    precision = score_dict['precision']
    recall    = score_dict['recall']
    f1        = score_dict['f1']

    score = 3 / (1/precision + 1/recall + 1/f1)

    # 5) 최대화 대상 점수 반환
    return float(score)

# === Optuna 실행(샘플러/프루너는 취향껏) ===
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=10, show_progress_bar=True)

print("BEST SCORE:", study.best_value)
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"  {k}: {v:.6f}")

# --- 결과를 텍스트 파일로 저장 ---
with open("best_result.txt", "w") as f:
    f.write(f"BEST SCORE: {study.best_value:.6f}\n")
    f.write("BEST PARAMS:\n")
    for k, v in study.best_params.items():
        # 숫자가 아닌 값일 수도 있으니 형식 안전하게 처리
        if isinstance(v, (int, float)):
            f.write(f"  {k}: {v:.6f}\n")
        else:
            f.write(f"  {k}: {v}\n")

print("\n✅ 결과가 'best_result.txt' 파일에 저장되었습니다.")