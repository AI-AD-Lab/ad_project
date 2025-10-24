import pandas as pd
from pathlib import Path
import os

from config import config
from _utils.plot_utils import plot_confusion_matrix_table
from rule_based_classification import excute_rule_based_classification
from _utils.score_utils import compute_score

if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / config['UNCLE_DIR_NAME']

    short_to_long_label = config['Short_to_Long_Label']
    label_data = pd.read_csv(SYN_LOG_DATA_ROOT_DIR / 'label.csv')
    labels = ['RA','ST', 'UT', 'LT', 'RT', 'LLC', 'RLC']

    confusion_matrix_save_dir = './output/plots/score/'
    confusion_matrix_png = confusion_matrix_save_dir + 'confusion_matrix.png'
    confusion_matrix_csv = confusion_matrix_save_dir + 'confusion_matrix.csv'

    df_total_result = excute_rule_based_classification(class_perm=labels)

    score_dict = compute_score(df_total_result)
    print(f"Precision: {score_dict['precision']:.4f}, Recall: {score_dict['recall']:.4f}, F1: {score_dict['f1']}")

    if not os.path.exists(confusion_matrix_save_dir):
        os.makedirs(confusion_matrix_save_dir)
    plot_confusion_matrix_table(df_total_result, save_path=confusion_matrix_png)
    df_total_result.to_csv(confusion_matrix_csv, index=False)
