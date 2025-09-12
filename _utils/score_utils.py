import pandas as pd
import numpy as np

def compute_score(conf_matrix: pd.DataFrame):
    """
    Args:
        conf_matrix: numpy 2D array (real x predicted)
    Returns:
        macro_precision, macro_recall
    """
    conf_matrix = conf_matrix.drop(columns=['TOTAL'])
    conf_matrix = conf_matrix.to_numpy()
    num_classes = conf_matrix.shape[0]

    precisions = []
    recalls = []

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    return {"precision": float(macro_precision), "recall": float(macro_recall), "f1": float(f1)}


