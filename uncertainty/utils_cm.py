import numpy as np


def compute_cm(gt_val, preds_val, classes):
    cm = np.zeros((len(classes), len(classes)))
    for c1 in classes:
        for c2 in classes:
            cm[c1, c2] = (preds_val[gt_val == c1] == c2).sum()
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    return np.round(recall, 2), np.round(precision, 2)


def split_dataset(data, labels, split):
    # skLearn could be used as well for easily splitting the data
    N = data.shape[0]
    split = [int(v * N) for v in split]
    rand_idx = np.random.permutation(N)
    data = data[rand_idx]
    labels = labels[rand_idx]
    if len(split) == 1:
        return data, labels
    split = split[:-1]
    if len(split) == 1:
        data_train, data_val = np.split(data, split)
        labels_train, labels_val = np.split(labels, split)
        return data_train, labels_train, data_val, labels_val
    elif len(split) == 2:
        data_train, data_val, data_test = np.split(data, split)
        labels_train, labels_val, labels_test = np.split(labels, split)
        return data_train, labels_train, data_val, labels_val, data_test, labels_test
    else:
        raise NotImplementedError