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
# SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
# labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']
# labels = ['ST','RA', 'UT', 'LT', 'RT', 'LLC', 'RLC']
labels = ['UT', 'RA', 'LT', 'RT', 'ST', 'LLC', 'RLC']


def optimiezd_classification(parameterset) -> pd.DataFrame:

    LLC_params = parameterset['llc_params']
    RLC_params = parameterset['rlc_params']
    LT_params = parameterset['lt_params']
    RT_params = parameterset['rt_params']
    RA_params = parameterset['ra_params']
    ST_params = parameterset['st_params']
    UT_params = parameterset['ut_params']

    class_perm = labels
    real_index = { short_to_long_label[label]:idx for idx, label in enumerate(class_perm) }

    # 준비: 행 인덱스(정답 라벨 → 행번호), 열 정의
    class_perm = labels
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
                if detect_left_lane_change(data, threshold=LLC_params['threshold'], duration_sec=LLC_params['duration_sec']):
                    return i
            elif lab == 'RLC':
                if detect_right_lane_change(data, threshold=RLC_params['threshold'], duration_sec=RLC_params['duration_sec']):
                    return i
            elif lab == 'ST':
                if detect_straight(data, abs_normal_threshold=ST_params['abs_normal_threshold'], abs_threshold=ST_params['abs_threshold'], duration_sec=ST_params['duration_sec']):
                    return i
            elif lab == 'RT':
                if detect_right_turn(data, right_threshold=RT_params['right_threshold'], 
                                     duration_sec=RT_params['duration_sec'],
                                     max_duration_sec=RT_params['max_duration_sec']):
                    return i
            elif lab == 'LT':
                if detect_left_turn(data, left_threshold=LT_params['left_threshold'], 
                                    duration_sec=LT_params['duration_sec'],
                                    max_duration_sec=LT_params['max_duration_sec']
                                    ):
                    return i
            elif lab == 'RA':
                if  detect_roundabout(data, threshold_neg=RA_params['threshold_neg'], threshold_pos=RA_params['threshold_pos'], duration_sec=RA_params['duration_sec'], max_duration_sec=RA_params['max_duration_sec'] ):
                    return i
            elif lab == 'UT':
                if detect_u_turn(data, threshold=UT_params['threshold'], duration_sec=UT_params['duration_sec']):
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
    # llc_threshold = trial.suggest_float("llc_threshold", 0.1, 0.4)
    # llc_duration  = trial.suggest_float("llc_duration_sec", 0.4, 0.9)

    llc_threshold =  0.212362
    llc_duration  = 0.875357
    # RLC
    rlc_threshold = 0.137831
    rlc_duration  = 0.833088

    # ST
    st_abs_norm_th  = 0.059045
    st_abs_th       = 0.109141
    st_duration     = 7.532441

    # RT
    rt_right_th = 1.183795
    rt_duration = 1.574704
    rt_max_duration  = 5.479960

    # LT (typo 주의: left_threshold)
    lt_left_th  = -0.427932
    lt_duration = 4.137019
    lt_max_duration  = 6.156221
    # lt_max_duration  = 8.0


    # RA
    ra_th_neg    = -0.146373
    ra_th_pos    = 0.097355
    ra_duration  = 1.454986
    ra_max_dur   = trial.suggest_float("ra_max_duration_sec", 13.0, 14.0)

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
            "duration_sec": lt_duration,
            "max_duration_sec":lt_max_duration
        },
        'rt_params':{
            "right_threshold": rt_right_th,
            "duration_sec": rt_duration,
            "max_duration_sec": rt_max_duration
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
    df_total_result = optimiezd_classification(parameterset)
    # 4) 점수 계산 (compute_score가 스칼라를 반환한다고 가정)
    score_dict = compute_score(df_total_result)
    precision = score_dict['precision']
    recall    = score_dict['recall']
    f1        = score_dict['f1']

    # print(f"Precision: {score_dict['precision']:.4f}, Recall: {score_dict['recall']:.4f}, F1: {score_dict['f1']}")
    # 5) 최대화 대상 점수 반환
    return f1

# === Optuna 실행(샘플러/프루너는 취향껏) ===
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=200, show_progress_bar=True)

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