import pandas as pd
from pathlib import Path
import os

from config import config
from _utils.plot_utils import plot_confusion_matrix_table
from rule_based_classification import optimiezd_classification
from _utils.score_utils import compute_score

if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    # SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

    short_to_long_label = config['Short_to_Long_Label']
    label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
    
    confusion_matrix_save_dir = './output/plots/score/'
    confusion_matrix_png = confusion_matrix_save_dir + 'confusion_matrix.png'
    confusion_matrix_csv = confusion_matrix_save_dir + 'confusion_matrix.csv'
    from time import time
    start_time = time()
    df_total_result = optimiezd_classification(class_perm=None)
    end_time = time()
    print(f"Total processing time: {end_time - start_time:.4f} seconds")
    print(f"Processing time per sample: {(end_time - start_time) / len(label_data):.4f} seconds")
    print(f"Samples per second: {len(label_data) / (end_time - start_time):.2f} samples/second")

    acc =0
    for col, row in zip(['RA', 'UT', 'LT', 'RT', 'ST', 'LLC', 'RLC'], [0,1,2,3,4,5,6]):
        t = config['Short_to_Long_Label'][col]
        acc +=  df_total_result[col][row]
    acc /= len(label_data)

    score_dict = compute_score(df_total_result)

    print(f"Acc: {acc:.4f} Precision: {score_dict['precision']:.4f}, Recall: {score_dict['recall']:.4f}, F1: {score_dict['f1']:.4f}")

    # if not os.path.exists(confusion_matrix_save_dir):
    #     os.makedirs(confusion_matrix_save_dir)
    # plot_confusion_matrix_table(df_total_result, save_path=confusion_matrix_png)
    # df_total_result.to_csv(confusion_matrix_csv, index=False)
