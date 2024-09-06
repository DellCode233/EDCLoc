from sklearn.metrics import (
    accuracy_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
import numpy as np


def calc_metric(y_true, y_score=None, y_pred=None, threshold=0.5):
    y_true = np.array(y_true, np.int64)
    y_pred = np.where(y_score > threshold, 1, 0) if y_pred is None and y_score is not None else y_pred
    myDict = {}
    matrix = confusion_matrix(y_true, y_pred)  # type: ignore
    TP, FN, FP, TN = matrix[[1, 1, 0, 0], [1, 0, 1, 0]]
    myDict["Sn"] = TP / (TP + FN + 1e-06)
    myDict["Sp"] = TN / (FP + TN + 1e-06)
    myDict["Acc"] = accuracy_score(y_true, y_pred)  # type: ignore
    myDict["Recall"] = recall_score(y_true, y_pred)  # type: ignore
    myDict["MCC"] = matthews_corrcoef(y_true, y_pred)  # type: ignore
    myDict["Precision"] = precision_score(y_true, y_pred)  # type: ignore
    myDict["F1"] = f1_score(y_true, y_pred)  # type: ignore
    if y_score is not None:
        myDict["AP"] = average_precision_score(y_true, y_score)  # type: ignore
        myDict["AUC"] = roc_auc_score(y_true, y_score)
    return myDict


def calc_confusion_matrix(y_true, y_score=None, y_pred=None, threshold=0.5):
    y_true = np.array(y_true, np.int64)
    y_pred = np.where(y_score > threshold, 1, 0) if y_pred is None and y_score is not None else y_pred
    myDict = {}
    matrix = confusion_matrix(y_true, y_pred)  # type: ignore
    TP, FN, FP, TN = matrix[[1, 1, 0, 0], [1, 0, 1, 0]]
    return dict(TP=TP, FN=FN, FP=FP, TN=TN)
