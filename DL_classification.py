#%%
import pandas as pd
from pathlib import Path
import numpy as np

from config import config

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image  # (이미지 데이터일 경우)
import os
import matplotlib.pyplot as plt
from _utils.data_processing_utils import data_load
import seaborn as sns
from brokenaxes import brokenaxes

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


class TrajectoryDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            csv_path (str or Path): label.csv 파일 경로
            data_root (str or Path): 데이터 파일들이 위치한 루트 디렉토리
            transform (callable, optional): 데이터 변환 함수 (이미지 전처리 등)
        """
        label_data = Path(data_root) / 'label.csv'
        self.df = pd.read_csv(label_data)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        label = row["trajectory type"]

        # 예시: 이미지 파일 로드
        file_path = self.data_root / file_name
        
        trajectory_data = data_load(file_path)
        position_vel_acc = trajectory_data[['PositionX (m)','PositionY (m)','PositionZ (m)',
                                            'VelocityX(EntityCoord) (km/h)','VelocityY(EntityCoord) (km/h)','VelocityZ(EntityCoord) (km/h)',
                                            'AccelerationX(EntityCoord) (m/s2)', 'AccelerationY(EntityCoord) (m/s2)', 'AccelerationZ(EntityCoord) (m/s2)'
                                            ]]
        # Tensor로 변환
        # position_vel_acc = torch.tensor(np.array(position_vel_acc), dtype=torch.long)
        label = torch.tensor(config['label_to_class'][label], dtype=torch.long)
        return position_vel_acc, label

def shrink_ts(y, target_len=900, method='interp', t=None, fs=None):
    """
    길이가 target_len보다 긴 시계열 y를 짧게 변환.
    - y: 1D array-like (numpy/pandas Series)
    - target_len: 최종 길이
    - method: 'interp' | 'avgpool' | 'downsample' | 'crop'
    - t: 시간축(초). 있다면 보간에 사용
    - fs: 샘플링 주파수(Hz). downsample 방법에서 안티앨리어싱 설정에 사용(선택)
    """
    y = np.asarray(y)

    if y.dtype == object:
        y = y.astype(float)
    
    n = len(y)
    if n <= target_len:
        return y  # 이미 짧거나 같으면 그대로

    if method == 'interp':
        # (1) 선형 보간으로 길이를 정확히 맞춤 — 모양을 최대한 보존
        if t is None:
            x = np.linspace(0.0, 1.0, n)
            x_new = np.linspace(0.0, 1.0, target_len)
        else:
            x = np.asarray(t)
            x_new = np.linspace(x.min(), x.max(), target_len)
        return np.interp(x_new, x, y)

    elif method == 'avgpool':
        # (2) 블록 평균(평균 풀링) — 노이즈 억제 + 길이 축소
        #    target_len으로 정확히 나누어떨어지도록 끝을 잘라내고 평균
        k = int(np.floor(n / target_len))  # 블록 크기
        y_cut = y[:k * target_len]
        return y_cut.reshape(target_len, k).mean(axis=1)

    elif method == 'downsample':
        # (3) 등간격 다운샘플링(간단) + (선택) 저역통과 권장
        #    scipy 없이 간단히 구현: 인덱스 골라뽑기
        idx = np.linspace(0, n - 1, target_len).astype(int)
        y_ds = y[idx]
        # 만약 스펙트럼 보존이 중요하고 scipy 가능하면:
        # from scipy.signal import resample  # FFT기반
        # y_ds = resample(y, target_len)
        return y_ds

    elif method == 'crop':
        # (4) 가운데 크롭 — 중심부만 사용 (패턴이 중앙에 몰려있을 때)
        start = (n - target_len) // 2
        return y[start:start + target_len]
    
    elif method == 'last_crop':
        # (5) 마지막 부분 추출 (부족하면 0으로 패딩)
        if n >= target_len:
            start = n - target_len
            return y[start:start + target_len]
        else:
            pad_len = target_len - n
            pad = np.zeros(pad_len, dtype=y.dtype)
            return np.concatenate([pad, y])
    else:
        raise ValueError("method must be one of {'interp','avgpool','downsample','crop', 'crop'}")

def draw(y):
    y = np.array(y) /50 
    bin_width = 1
    bins = np.arange(0, y.max() + bin_width, bin_width)

    # 2️⃣ 히스토그램 계산
    counts, edges = np.histogram(y, bins=bins)

    # 3️⃣ 막대그래프 시각화
    plt.figure(figsize=(10, 5))
    plt.bar(edges[:-1], counts, width=bin_width * 0.9, color='royalblue', alpha=0.7, edgecolor='black')

    # 4️⃣ 그래프 꾸미기
    # plt.title("Distribution of Sequence Lengths", fontsize=14, fontweight='bold')
    plt.xlabel("Length (second)", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.4)

    # X축 눈금 정리
    plt.xticks(np.arange(0, y.max(), 10))

    plt.tight_layout()
    plt.show()


def draw_2(y):
    y = np.array(y) / 50
    bin_width = 2
    bins = np.arange(0, y.max() + bin_width, bin_width)
    counts, edges = np.histogram(y, bins=bins)

    # broken axis 설정 (예: 0~60, 120~190만 표시)
    bax = brokenaxes(xlims=((0, 55), (140, 185)), hspace=0.05, despine=False, fig=plt.figure(figsize=(10,5)))

    bax.bar(edges[:-1], counts, width=bin_width * 0.9 , color='royalblue', alpha=0.7, edgecolor='black')
    bax.set_xlabel("Length (second)", fontsize=20)
    bax.set_ylabel("Count", fontsize=20)

    plt.tight_layout()
    plt.show()

def _add_vertical_wiggle(ax, side='right', where='both', char='≈', size=13, pad=0.012):
    # side: 'right'는 왼쪽 축의 오른쪽 경계, 'left'는 오른쪽 축의 왼쪽 경계
    x = 1.0 if side == 'right' else 0.0
    x = x + pad if side == 'right' else x - pad
    ys = [0.0, 1.0] if where == 'both' else ([0.0] if where == 'bottom' else [1.0])
    for y in ys:
        ax.text(x, y, char, rotation=90, ha='center', va='center',
                transform=ax.transAxes, fontsize=size, weight='bold',
                clip_on=False)

def draw_pretty_broken_hist(y):
    y = np.array(y) / 50
    bin_width = 2
    bins = np.arange(0, y.max() + bin_width, bin_width)
    counts, edges = np.histogram(y, bins=bins)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(8, 3.8),
        gridspec_kw={'wspace': 0.06, 'width_ratios': [3, 2]}
    )

    # 왼/오 구간
    for ax in (ax1, ax2):
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)

    # 히스토그램
    color = '#1f77b4'
    ax1.bar(edges[:-1], counts, width=bin_width*0.85,
            color=color, alpha=0.8, edgecolor='black', linewidth=0.6)
    ax2.bar(edges[:-1], counts, width=bin_width*0.85,
            color=color, alpha=0.8, edgecolor='black', linewidth=0.6)

    ax1.set_xlim(0, 55)
    ax2.set_xlim(145, 185)

    # ❶ 가운데 경계선 없애기
    ax1.spines['right'].set_visible(False)  # ← 중앙선 제거
    ax2.spines['left'].set_visible(False)   # ← 중앙선 제거
    ax2.tick_params(labelleft=False)        # 오른쪽 축의 왼쪽 눈금 제거

    # ❷ “사이”의 위/아래에 각각 세로 물결표 1개만 넣기
    #    (두 축의 실제 위치를 이용해 정확히 가운데에 배치)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    cx = (pos1.x1 + pos2.x0)/2.0 + 0.002          # 두 축 사이 중앙 x (figure 좌표)
    top_y    = pos1.y1               # 위쪽 살짝 바깥
    bottom_y = pos1.y0

    from matplotlib.patches import Rectangle
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    gap_x = pos1.x1
    gap_w = (pos2.x0 - pos1.x1)
    offset = 0.002
    # 공백 영역을 흰색 직사각형으로 덮기 (격자 잔상 완전히 제거)
    fig.add_artist(Rectangle((gap_x + offset, 0.2), gap_w, 0.65,
                            transform=fig.transFigure,
                            facecolor='white', edgecolor='none',
                            zorder=5))
    # 아래쪽 살짝 바깥
    # fig.text(cx, top_y,    '≈', rotation=90, ha='center', va='center',
    #          fontsize=12, weight='bold', clip_on=False)
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

def get_velocity(data:pd.DataFrame):
    vel_cols = [
        'VelocityX(EntityCoord) (km/h)',
        'VelocityY(EntityCoord) (km/h)',
        'VelocityZ(EntityCoord) (km/h)'
    ]

    # 속도 크기 계산 (벡터의 크기)
    data['Speed'] = np.sqrt(
        data[vel_cols[0]]**2 +
        data[vel_cols[1]]**2 +
        data[vel_cols[2]]**2
    )

    return data

def correlation_analysis(df, method='pearson', figsize=(8, 6), cmap='coolwarm', title=None):
    """
    데이터 상관관계 분석 및 시각화

    Parameters:
    ----------
    df : pd.DataFrame  
        분석할 데이터프레임 (숫자형 컬럼만 분석)
    method : str, default='pearson'  
        상관계수 계산 방식 ('pearson', 'spearman', 'kendall')
    figsize : tuple, default=(8, 6)  
        그래프 크기
    cmap : str, default='coolwarm'  
        히트맵 색상
    title : str, optional  
        그래프 제목 (미지정 시 자동 생성)
    
    Returns:
    -------
    corr_df : pd.DataFrame  
        상관계수 행렬
    """

    # 1️⃣ 숫자형 컬럼만 선택
    df_numeric = df.select_dtypes(include=[np.number])

    if df_numeric.empty:
        raise ValueError("⚠️ 숫자형 컬럼이 없습니다. 상관관계 분석 불가.")

    # 2️⃣ 상관계수 계산
    corr_df = df_numeric.corr(method=method)

    # 3️⃣ 시각화
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        annot=True,         # 상관계수 숫자 표시
        fmt=".2f",          # 소수점 2자리
        cmap=cmap,
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    # 4️⃣ Times New Roman 스타일 적용 (선택)
    plt.title(title or f"Correlation Matrix ({method.title()})", fontsize=14, fontweight='bold', fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman', fontsize=10, rotation=45)
    plt.yticks(fontname='Times New Roman', fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    return corr_df


if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    # SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'
    
    confusion_matrix_save_dir = './output/plots/score/'
    dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR)

    total_len = []
    for data, idx in dataset:
        # data = shrink_ts(data, method='last_crop')
        # correlation_analysis(data)
        # break
        # print(len(data))
        total_len.append(len(data))

    # 길이가 2000 이상인 샘플
    # count  =  sum(1 for length in total_len if length >= 2000)
    # print(f'Length >= 2000 : {count} samples / {count/len(total_len)*100:.2f}%')
    # draw_2(total_len) 
    draw_pretty_broken_hist(total_len)
    # key_scenario_statelog_dir = GRANDPARENTS_DIR / 'trajectory_stratelog_with_velocity'
    # csv_files = os.listdir(key_scenario_statelog_dir)
    # information_csv = pd.read_csv(key_scenario_statelog_dir / 'information.csv')
    
    # for t, file in information_csv.groupby('type'):
    #     print(t)
    #     total = None
    #     for f in file['file_name']:
            
    #         data = pd.read_csv(key_scenario_statelog_dir/f)
    #         data = get_velocity(data)
    #         data = data[[
    #             'Speed',
    #             'AccelerationX(EntityCoord) (m/s2)',
    #             'AccelerationY(EntityCoord) (m/s2)',
    #             'RotationZ (deg)',
    #             'FrontWheelAngle (deg)',

    #         ]]
    #         # 1️⃣ 숫자형 컬럼만 선택
    #         df_numeric = data.select_dtypes(include=[np.number])
    #         # 2️⃣ 상관계수 계산
    #         corr = df_numeric.corr(method='pearson')

    #         if total is None:
    #             total = corr
    #         else:
    #             total += corr

    #     total /= len(file)

    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(
    #         total,
    #         annot=True,         # 상관계수 숫자 표시
    #         fmt=".2f",          # 소수점 2자리
    #         cmap='coolwarm',
    #         vmin=-1, vmax=1,
    #         square=True,
    #         cbar_kws={"shrink": 0.8}
    #     )

    #     # 4️⃣ Times New Roman 스타일 적용 (선택)
    #     plt.title(f"Correlation Matrix", fontsize=14, fontweight='bold', fontname='Times New Roman')
    #     plt.xticks(fontname='Times New Roman', fontsize=10, rotation=45)
    #     plt.yticks(fontname='Times New Roman', fontsize=10, rotation=0)
    #     plt.tight_layout()
    #     plt.savefig(f'{key_scenario_statelog_dir/t}_confusion.png' )
    #     plt.show()
    #     plt.close()


# %%
