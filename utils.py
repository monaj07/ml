import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(labels_test, percept_preds):
    cm = confusion_matrix(labels_test, percept_preds)
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    F1_percept = 2 * recall.mean() * precision.mean() / (recall.mean() + precision.mean())
    return cm, F1_percept