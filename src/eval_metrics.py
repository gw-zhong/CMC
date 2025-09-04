import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_sims_classification(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # three classes
    y_pred_3 = np.argmax(y_pred, axis=1)
    Mult_acc_3 = accuracy_score(y_pred_3, y_true)
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
    # two classes
    y_pred = np.array([[v[0], v[2]] for v in y_pred])
    # with 0 (<= 0 or > 0) **NOTE: Different from MOSI!** SIMS: non-positive / positive
    y_pred_2 = np.argmax(y_pred, axis=1)
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v <= 1 else 1)
    y_true_2 = np.array(y_true_2)
    Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
    # without 0 (< 0 or > 0)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = y_pred[non_zeros]
    y_pred_2 = np.argmax(y_pred_2, axis=1)
    y_true_2 = y_true[non_zeros]
    Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    eval_results = {
        "Has0_acc_2":  round(Has0_acc_2, 4),
        "Has0_F1_score": round(Has0_F1_score, 4),
        "Non0_acc_2":  round(Non0_acc_2, 4),
        "Non0_F1_score": round(Non0_F1_score, 4),
        "Acc_3": round(Mult_acc_3, 4),
        "F1_score_3": round(F1_score_3, 4)
    }
    return eval_results

def eval_mosi_classification(y_pred, y_true):
    """
    {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    # three classes
    y_pred_3 = np.argmax(y_pred, axis=1)
    Mult_acc_3 = accuracy_score(y_pred_3, y_true)
    F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
    # two classes
    y_pred = np.array([[v[0], v[2]] for v in y_pred])
    # with 0 (< 0 or >= 0) **NOTE: Different from SIMS!** MOSI/MOSEI: negative / non-negative
    y_pred_2 = np.argmax(y_pred, axis=1)
    y_true_2 = []
    for v in y_true:
        y_true_2.append(0 if v < 1 else 1)
    y_true_2 = np.array(y_true_2)
    Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
    # without 0 (< 0 or > 0)
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
    y_pred_2 = y_pred[non_zeros]
    y_pred_2 = np.argmax(y_pred_2, axis=1)
    y_true_2 = y_true[non_zeros]
    Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
    Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

    eval_results = {
        "Has0_acc_2": round(Has0_acc_2, 4),
        "Has0_F1_score": round(Has0_F1_score, 4),
        "Non0_acc_2": round(Non0_acc_2, 4),
        "Non0_F1_score": round(Non0_F1_score, 4),
        "Acc_3": round(Mult_acc_3, 4),
        "F1_score_3": round(F1_score_3, 4)
    }
    return eval_results