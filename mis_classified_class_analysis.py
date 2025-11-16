#%%
import os
import pandas as pd
import numpy as np

from _utils.data_processing_utils import data_load

from rule_utils.lane_change import detect_right_lane_change, detect_left_lane_change
from rule_utils.straight import detect_straight
from rule_utils.left_turn import detect_left_turn
from rule_utils.right_turn import detect_right_turn
from rule_utils.roundabout import detect_roundabout
from rule_utils.u_turn import detect_u_turn

from pathlib import Path
from config import config
from rule_utils_masking.common_util import rolling 
import time
import matplotlib.pyplot as plt
import numpy as np

from rule_utils_masking.lane_change import (
    detect_left_lane_change_mask,
    detect_right_lane_change_mask,
)
from rule_utils_masking.straight import detect_straight_mask, detect_straight_mask_flexible
from rule_utils_masking.right_turn import detect_right_turn_mask
from rule_utils_masking.left_turn import detect_left_turn_mask

from rule_utils_masking.roundabout import detect_roundabout_mask
from rule_utils_masking.u_turn import detect_u_turn_mask

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


#CONFIG
GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

short_to_long_label = config['Short_to_Long_Label']
label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
labels = ['UT', 'RA', 'LT', 'RT', 'ST', 'LLC', 'RLC']
# labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']

optimiezd_rule_parameters = { 
    'LLC': {'duration_sec':0.875357, 'threshold':0.212362},
    'RLC': {'duration_sec':0.833088, 'threshold':0.137831}, 
    'ST': {'abs_normal_threshold':0.059045, 'abs_threshold':0.109141, 'duration_sec':7.532441}, 
    'RT': {'right_threshold':1.183795, 'duration_sec':1.574704, 'max_duration_sec':5.479960}, 
    'LT': {'left_threshold':-0.427932, 'duration_sec':4.137019, 'max_duration_sec':6.156221}, 
    'RA': {'threshold_neg':-0.146373, 'threshold_pos':0.097355, 'duration_sec':1.454986, 'max_duration_sec':14.449369}, 
    'UT': {'threshold':2.187467, 'duration_sec':2.823866} }

def miss_classied_finder() -> pd.DataFrame:
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

    llc_param = optimiezd_rule_parameters['LLC']
    rlc_param = optimiezd_rule_parameters['RLC']
    st_param = optimiezd_rule_parameters['ST']
    rt_param = optimiezd_rule_parameters['RT']
    lt_param = optimiezd_rule_parameters['LT']
    ra_param = optimiezd_rule_parameters['RA']
    ut_param = optimiezd_rule_parameters['UT']

    # detect 함수 디스패처 (순서대로 평가 → True면 즉시 중단)
    def eval_in_order(data):
        # class_perm 순서대로 필요할 때만 평가
        for i, lab in enumerate(perm_array):
            if lab == 'LLC':
                if detect_left_lane_change(data, duration_sec=llc_param['duration_sec'], threshold=llc_param['threshold']):
                    return i
            elif lab == 'RLC':
                if detect_right_lane_change(data, duration_sec=rlc_param['duration_sec'], threshold=rlc_param['threshold']):
                    return i
            elif lab == 'ST':
                if detect_straight(data, abs_normal_threshold= st_param['abs_normal_threshold'],
                                    abs_threshold=st_param['abs_threshold'], 
                                    duration_sec=st_param['duration_sec']):
                    return i
            elif lab == 'RT':
                if detect_right_turn(data, right_threshold=rt_param['right_threshold'], 
                                     duration_sec=rt_param['duration_sec'],
                                     max_duration_sec=rt_param["max_duration_sec"]
                                     ):
                    return i
            elif lab == 'LT':
                if detect_left_turn(data, left_threshold=lt_param['left_threshold'], 
                                    duration_sec=lt_param['duration_sec'],
                                    max_duration_sec=lt_param["max_duration_sec"]):
                    return i
            elif lab == 'RA':
                if detect_roundabout(data, threshold_neg=ra_param['threshold_neg'],
                                     threshold_pos=ra_param['threshold_pos'],
                                     duration_sec=ra_param['duration_sec'], 
                                     max_duration_sec=ra_param['max_duration_sec']):
                    return i
            elif lab == 'UT':
                if detect_u_turn(data, threshold=ut_param['threshold'], 
                                 duration_sec=ut_param['duration_sec']):
                    return i
        return None  # 아무 클래스에도 해당 안 됨

    miss_classed_files_list = []
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

        if pred_idx is None:
            pred_idx = 7

        if row_idx != pred_idx:
            # print(label_idx, predicted_idx)
            miss_classed_files_list.append([file, cols[row_idx], cols[pred_idx]])

    return pd.DataFrame(miss_classed_files_list, columns=['file_name', 'true_label', 'predicted_label'])

# --- 라벨 → (mask_func, 필요한 파라미터 키 목록) 매핑 ---
LABEL_DISPATCH = {
    "LLC": (detect_left_lane_change_mask,  ["duration_sec", "threshold"]),
    "RLC": (detect_right_lane_change_mask, ["duration_sec", "threshold"]),
    # "ST":  (detect_straight_mask_flexible,          ["abs_normal_threshold", "abs_threshold", "duration_sec"]),
    "ST":  (detect_straight_mask,          ["abs_normal_threshold", "abs_threshold", "duration_sec"]),
    "RT":  (detect_right_turn_mask,        ["right_threshold", "duration_sec" ,"max_duration_sec"]),
    "LT":  (detect_left_turn_mask,         ["left_threshold", "duration_sec", "max_duration_sec"]),
    "RA":  (detect_roundabout_mask,        ["threshold_neg", "threshold_pos", "duration_sec", "max_duration_sec"]),
    "UT":  (detect_u_turn_mask,            ["threshold", "duration_sec"]),
}

def _compute_mask_by_label(data, label: str, params: dict) -> np.ndarray:
    """
    라벨에 맞는 마스크 함수를 찾아 data에 적용.
    params 딕셔너리에서 해당 라벨에 필요한 키만 꺼내 kwargs로 전달.
    """
    if label not in LABEL_DISPATCH:
        raise ValueError(f"지원하지 않는 라벨입니다: {label}")

    func, needed_keys = LABEL_DISPATCH[label]
    kwargs = {k: params[k] for k in needed_keys if k in params}
    # 공통적으로 쓸 수 있는 기본 인자(컬럼명/롤링 설정 등)도 넘기고 싶다면 여기서 추가하세요.
    return func(data, **kwargs)


def plot_masked_comparison(
    data,
    true_label: str,
    predicted_label: str,
    optimized_rule_parameters: dict,
    time_col: str = "time (sec)",
    y_col: str = "AccelerationY(EntityCoord) (m/s2)",
    line_label: str = "Lateral Acceleration(m/s²)",
    save_path: str | None = None
):
    """
    True/Predicted 구간 마스크를 시각화합니다.
    - True 라벨 → 빨간색
    - Predicted 라벨 → 파란색
    - 둘 다 True → 보라색
    """
    # 1) 축 데이터
    t = data[time_col].to_numpy()

    # y축 데이터 rolling
    y = rolling(data, rolling_seconds=2.0, sampling_hz=50)[y_col].to_numpy()

    # 2) 라벨별 파라미터 체크
    if true_label not in optimized_rule_parameters:
        raise KeyError(f"optimized_rule_parameters에 '{true_label}' 키가 없습니다.")
    if predicted_label not in optimized_rule_parameters:
        raise KeyError(f"optimized_rule_parameters에 '{predicted_label}' 키가 없습니다.")

    # 3) 마스크 계산 (라벨에 따라 자동 디스패치)
    mask_true = _compute_mask_by_label(data, true_label, optimized_rule_parameters[true_label])
    mask_pred = _compute_mask_by_label(data, predicted_label, optimized_rule_parameters[predicted_label])

    if len(mask_true) != len(t) or len(mask_pred) != len(t):
        raise ValueError("마스크 길이와 시간축 길이가 일치하지 않습니다. 마스크 함수의 출력 길이를 확인하세요.")

    # 4) 겹치는 부분
    mask_overlap = mask_true & mask_pred
    mask_true_only = mask_true & ~mask_pred
    mask_pred_only = mask_pred & ~mask_true

    # 5) 플롯
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, color="black", lw=1.2, label=line_label )

    # 채우기 순서(겹침을 맨 위에 그려 가독성↑)
    plt.fill_between(t, y.min(), y.max(), where=mask_true_only, color="red",   alpha=0.25, label=f"True: {true_label}")
    plt.fill_between(t, y.min(), y.max(), where=mask_pred_only, color="blue",  alpha=0.25, label=f"Pred: {predicted_label}")
    plt.fill_between(t, y.min(), y.max(), where=mask_overlap,   color="purple",alpha=0.40, label="Overlap")

    plt.xlabel("Time(sec)", fontname="Times New Roman",fontsize=20, fontweight="bold"  )
    plt.ylabel('Lateral Acceleration(m/s²)',  fontname="Times New Roman",fontsize=20, fontweight="bold" )
    plt.legend(
        prop={
            'family': 'Times New Roman',  # 글씨체
            'size': 12,                   # 글자 크기
            'weight': 'bold'              # 볼드체
        },
        title_fontproperties={
            'family': 'Times New Roman',
            'size': 12,
            'weight': 'bold'
        },
        loc='upper right'
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

#%%
import time
miss_classed_df = miss_classied_finder()
miss_classed_df.to_csv('./output/mis_classified_class_analysis_final.csv', index=False)
#%%
for item in miss_classed_df.values:
    file_path = SYN_LOG_DATA_ROOT_DIR / item[0]
    true_label = item[1]
    predicted_label = item[2]

    data = data_load(file_path)

    plot_masked_comparison(
        data=data,
        true_label=true_label,
        predicted_label=predicted_label,
        optimized_rule_parameters=optimiezd_rule_parameters,
        time_col="time (sec)",
        y_col="AccelerationY(EntityCoord) (m/s2)",
        save_path = f'./output/mis_classified_plots_final/{item[0].replace("/", "_").replace(".csv", "")}_true_{true_label}_pred_{predicted_label}.png'
    )


# %%
