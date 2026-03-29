import numpy as np
import torch
from sklearn.metrics import auc, f1_score, precision_recall_fscore_support


def f1_triangular(gt: torch.Tensor, pred: torch.Tensor):
    """
    sklearnのf1_scoreを用いて二次構造行列の上三角行列（対角成分を除く）に対してF1スコアを計算する関数.
    多分使わない.
    
    Args:        
        gt (torch.Tensor): 真の二次構造行列. shape = (L, L)
        pred (torch.Tensor): 予測された二次構造行列. shape = (L, L)
    
    Returns:
        float: F1スコア
    """
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(gt.shape[0], gt.shape[1], offset=1)

    gt = gt[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(gt, pred, zero_division=0)


def f1_strict(gt: torch.Tensor, pred: torch.Tensor):
    """
    sklearnのprecision_recall_fscore_supportを用いて二次構造行列の上三角行列（対角成分を除く）に対してF1スコアを計算する関数.
    Args:
        gt (torch.Tensor): 真の二次構造行列. shape = (L, L)
        pred (torch.Tensor): 予測された二次構造行列. shape = (L, L)
    Returns:
        tuple[float, float, float]: precision, recall, F1スコア
    """
    
    # corner case when there are no positives
    # if len(ref_bp) == 0 and len(pre_bp) == 0:
    #     return 1.0, 1.0, 1.0

    # tp1 = 0
    # for rbp in ref_bp:
    #     if rbp in pre_bp:
    #         tp1 = tp1 + 1
    # tp2 = 0
    # for pbp in pre_bp:
    #     if pbp in ref_bp:
    #         tp2 = tp2 + 1

    # fn = len(ref_bp) - tp1
    # fp = len(pre_bp) - tp1

    # tpr = pre = f1 = 0.0
    # if tp1 + fn > 0:
    #     tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    # if tp1 + fp > 0:
    #     pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    # if tpr + pre > 0:
    #     f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    
    ind = torch.triu_indices(gt.shape[0], gt.shape[1], offset=1)

    gt = gt[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    if np.all(gt == 0) and np.all(pred == 0):
        return 1.0, 1.0, 1.0

    pre, rec, f1, support = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)

    return pre, rec, f1


def update_confusion_matrix(gt_bp_matrix: torch.Tensor, pred_bp_matrix: torch.Tensor, step: float, confusion_dict: dict):
    """
    混合行列 (tp, tn, fp, fn)を更新していく関数.

    Args:
        gt_bp_matrix (torch.Tensor): 真の二次構造行列. shape = (L, L)
        pred_bp_matrix (torch.Tensor): 予測された二次構造行列. shape = (L, L)
        step (float): 閾値のステップサイズ
        confusion_dict (dict): tp, tn, fp, fn を格納する辞書
    
    Returns:
        dict: tp, tn, fp, fn を格納する辞書. 各要素はサイズlen(thresholds)のnumpy配列
    """

    thresholds = np.arange(0, 1+step, step)

    gt_bp_matrix = gt_bp_matrix.view(-1).unsqueeze(1).numpy()   # (L, L) -> (L*L, 1)
    pred_bp_matrix = pred_bp_matrix.view(-1).unsqueeze(1).numpy() # (L, L) -> (L*L, 1)

    pred_bp_matrix = np.greater_equal(pred_bp_matrix, thresholds) # (L*L, 1) -> (L*L, len(thresholds))
    
    # 各閾値に対して混合行列を更新. 
    confusion_dict["tp"] += np.sum(np.logical_and(pred_bp_matrix, gt_bp_matrix), axis=0)
    confusion_dict["tn"] += np.sum(np.logical_and(np.logical_not(pred_bp_matrix), np.logical_not(gt_bp_matrix)), axis=0)
    confusion_dict["fp"] += np.sum(np.logical_and(pred_bp_matrix, np.logical_not(gt_bp_matrix)), axis=0)
    confusion_dict["fn"] += np.sum(np.logical_and(np.logical_not(pred_bp_matrix), gt_bp_matrix), axis=0)

    return confusion_dict

def calculate_auc(confusion_dict: dict):
    """
    混合行列からROC AUCとPR AUCを計算する関数.

    Args:
        confusion_dict (dict): tp, tn, fp, fn を格納する辞書. 各要素はサイズlen(thresholds)のnumpy配列
    
    Returns:
        tuple[dict, dict]: ROC曲線とPR曲線の図示するための材料を格納する辞書と, ROC AUCとPR AUCを格納する辞書
    """

    tp = confusion_dict["tp"]
    tn = confusion_dict["tn"]
    fp = confusion_dict["fp"]
    fn = confusion_dict["fn"]

    pre = tp / (tp + fp).astype(float)  # precision
    rec = tp / (tp + fn).astype(float)  # recall (true positive rate)
    fpr = fp / (tn + fp).astype(float)  # false positive rate

    pre[np.isnan(pre)] = 1

    roc_auc = auc(fpr, rec)
    pr_auc = auc(rec, pre)

    fig_materials = {
        "fpr": fpr,
        "pre": pre,
        "rec": rec,
    }

    aucs = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return fig_materials, aucs