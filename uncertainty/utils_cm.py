import numpy as np


def compute_cm(gt_val, preds_val, classes):
    cm = np.zeros((len(classes), len(classes)))
    for c1 in classes:
        for c2 in classes:
            cm[c1, c2] = (preds_val[gt_val == c1] == c2).sum()
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    return np.round(recall, 2), np.round(precision, 2)


def split_dataset(data, labels, Ts):
    N = data.shape[0]
    Ts = [int(v * N) for v in Ts]
    rand_idx = np.random.permutation(N)
    data = data[rand_idx]
    labels = labels[rand_idx]
    data_train = data[:Ts[0], :]
    labels_train = labels[:Ts[0]]
    data_val = data[Ts[0]:Ts[1], :]
    labels_val = labels[Ts[0]:Ts[1]]
    data_test = data[Ts[1]:, :]
    labels_test = labels[Ts[1]:]
    return data_train, labels_train, data_val, labels_val, data_test, labels_test