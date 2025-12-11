#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os

from config import config

from _utils.data_processing_utils import data_load
import matplotlib.pyplot as plt
# 전역 폰트 설정
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 예시 그래프
plt.plot([0, 1, 2, 3], [0, 10, 20, 50], color='royalblue')
plt.xlabel("Time (sec)", fontsize=12, fontweight='bold')
plt.ylabel(r"Lateral Acceleration (m/s$^2$)", fontsize=12, fontweight='bold')
#%%
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

if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

    short_to_long_label = config['Short_to_Long_Label']
    label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
    
    confusion_matrix_save_dir = './output/plots/score/'

    state_log_csv_files = [file for file in os.listdir(SYN_LOG_DATA_ROOT_DIR) if file.endswith("statelog.csv")]
    trajectory_types = set([cls_type.split("_")[-3] for cls_type in state_log_csv_files])
    
    cls_gatter = {}
    for trajectory_type in trajectory_types:
        for file in state_log_csv_files:
            if file.split("_")[-3] == trajectory_type:
                
                cls = file.split("_")[-3]
                if trajectory_type == '1' and (file.split("_")[-4] in ['RA3', 'RA9', 'RA12']):
                    cls = file.split("_")[-4]

                if cls not in cls_gatter.keys():
                    cls_gatter[cls] = [file]
                else:
                    cls_gatter[cls].append(file)

    # 궤적 종류, 파일명, 최대속도 csv화
    type_file_vel = []
    for cls in cls_gatter:
        for type_file in cls_gatter[cls]:
            file_path = SYN_LOG_DATA_ROOT_DIR / type_file
            data = pd.read_csv(file_path)
            data = data.drop(columns=['Entity'])
            
            vel_data = get_velocity(data)
            max_vel_raw = vel_data['Speed'].max()
            max_vel = int(round(max_vel_raw / 5.0) * 5)
            type_file_vel.append([cls, type_file, max_vel])
    
    save_dir = GRANDPARENTS_DIR / 'trajectory_stratelog_with_velocity'
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)

    df_type_file_vel = pd.DataFrame(type_file_vel, columns=['type', 'file_name', 'speed'])
    unique_df = df_type_file_vel.drop_duplicates(subset=['type', 'speed'], keep='first')
    unique_files = unique_df['file_name']
    unique_df.to_csv(save_dir/'information.csv')

    # import shutil
    # for file in unique_files:
    #     src = SYN_LOG_DATA_ROOT_DIR / file
    #     dst = save_dir / file
    #     shutil.copy(src, dst)

    #%%
    import matplotlib.pyplot as plt


    # type별로 묶어서 반복 (LLC1, RLC1 등)
    for t, group in unique_df.groupby('type'):
        plt.figure(figsize=(10, 6))
        print(f"==> Processing type: {t}")

        # 1) 속도 값 범위로 colormap 매핑 준비 (type 내부에서 속도별 일관된 색)
        speeds = sorted(group['speed'].unique())
        norm = plt.Normalize(min(speeds), max(speeds))
        cmap = plt.cm.plasma  # viridis/turbo 등 취향에 맞게

        # 이미 라벨 추가된 speed는 다시 legend에 넣지 않도록 관리
        labeled_speeds = set()

        for _, row in group.iterrows():
            file_path = save_dir / row['file_name']

            # CSV 파일 읽기
            try:
                temp = pd.read_csv(file_path)
            except Exception as e:
                print(f"⚠️ 파일 읽기 실패: {file_path} ({e})")
                continue

            # 필요 시 속도/시간 계산 함수 (사용하시던 함수)
            temp = get_velocity(temp).drop(columns=['Entity'])
            temp = temp.rolling(window=100, min_periods=1).mean()

            # 컬럼 체크 (실제 사용하는 컬럼 기준)
            if ('time (sec)' not in temp.columns) or ('AccelerationY(EntityCoord) (m/s2)' not in temp.columns):
                print(f"⚠️ 필요한 컬럼이 없습니다: {file_path}")
                continue

            # 2) 속도를 색으로 매핑
            spd = row['speed']
            color = cmap(norm(spd))

            # 3) 동일 속도의 라벨은 한 번만 legend에 올리기
            label = f"{spd} km/h" if spd not in labeled_speeds else None

            # plt.plot(
            #     temp['time (sec)'],
            #     temp['AccelerationY(EntityCoord) (m/s2)'],
            #     color=color,
            #     label=label 
            # )

            # plt.plot(
            #     temp['time (sec)'],
            #     temp['AccelerationX(EntityCoord) (m/s2)'],
            #     color=color,
            #     label=label 
            # )
            
            plt.plot(
                temp['time (sec)'],
                temp['Speed'],
                color=color,
                label=label 
            )

            labeled_speeds.add(spd)

        # --- 그래프 설정 ---
        plt.xlabel("Time(sec)",
            fontname="Times New Roman",   # 폰트
            fontsize=20,                  # 글씨 크기
            fontweight="bold"             # 굵게 (bold)
        )

        # plt.ylabel(
        #     "Lateral Acceleration(m/s²)",
        #     fontname="Times New Roman",   # 폰트
        #     fontsize=20,                  # 글씨 크기
        #     fontweight="bold"             # 굵게 (bold)
        # )

        plt.ylabel(
            "Velocity(km/h)",
            fontname="Times New Roman",   # 폰트
            fontsize=20,                  # 글씨 크기
            fontweight="bold"             # 굵게 (bold)
        )

        # 4) 우상단 범례: 속도별 색상 표시
        plt.legend(title="Speed", loc='upper right',
                    prop={
                        'family': 'Times New Roman',  # 글씨체
                        'size': 12,                   # 글자 크기
                        'weight': 'bold'              # 볼드체
                    },
                    title_fontproperties={
                        'family': 'Times New Roman', 
                        'size': 12, 
                        'weight': 'bold'
                    })

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir/t}_vel.png')
        # plt.show()
        plt.close()



# %%
