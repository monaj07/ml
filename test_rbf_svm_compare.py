#
import numpy as np
import matplotlib.pyplot as plt
from ml.classification import Perceptron, Fisher, LogisticRegression as Logistic, SVM
import logging as log
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

log.basicConfig(level=log.INFO)


def accuracy(labels_test, percept_preds):
    cm = confusion_matrix(labels_test, percept_preds)
    recall = np.diag(cm) / cm.sum(1)
    precision = np.diag(cm) / cm.sum(0)
    F1_percept = 2 * recall.mean() * precision.mean() / (recall.mean() + precision.mean())
    return cm, F1_percept


### Generate synthetic data using Gaussians
# class1:
size1 = 40
mean1 = np.array([-4, 3])
cov1 = np.array([[9, -3], [-3, 8]])
data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=size1)

size11 = 40
mean11 = np.array([35, 14])
cov11 = np.array([[9, -3], [-3, 8]])
data11 = np.random.multivariate_normal(mean=mean11, cov=cov11, size=size11)
data1 = np.vstack([data1, data11])

# class2:
size2 = 6000
mean2 = np.array([9, -2])
cov2 = np.array([[7, 6], [6, 7]])
data2 = np.random.multivariate_normal(mean2, cov2, size2)
# # class3:
# mean3 = np.array([3, 5])
# cov3 = np.array([[2, 0], [0, 2]])
# data3 = np.random.multivariate_normal(mean3, cov3, 150)

# Integrate the data from different classes and shuffle them
data = np.vstack([data1, data2])
labels = np.vstack([np.ones((data1.shape[0], 1)), -np.ones((data2.shape[0], 1))]).squeeze()
permutation = np.random.permutation(labels.size)
data = data[permutation, :]
labels = labels[permutation]
Ts = 0.6
data_train = data[:int(Ts * data.shape[0]), :]
labels_train = labels[:int(Ts * data.shape[0])]
data_test = data[int(Ts * data.shape[0]):, :]
labels_test = labels[int(Ts * data.shape[0]):]

# ----------------------------------------------------------------------------------------------------
# Instantiate the Perceptron classifier
# ----------------------------------------------------------------------------------------------------
percept = Perceptron((data_train, labels_train), learning_rate=0.1, n_iters=data.shape[0], reg=0.015)#, class_weight='class_mass')
percept.fit_transform()
# log.info(f"Trained linear classifier (No regularization + uniform class weights): w = {percept.w.T}")
percept_preds = percept.predict(data_test.copy())
cm, F1_percept = accuracy(labels_test, percept_preds)
print(f"Perceptron_F1 = {F1_percept}")
print(cm)
# ----------------------------------------------------------------------------------------------------

# --------------------------------------------------
# Instantiate the Perceptron classifier with outliers
# --------------------------------------------------
# picks = np.random.choice(data_train.shape[0], 500, replace=False)
# labels_train[picks] = -labels_train[picks]  # Change true labels of some randomly picked data points
# --------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Instantiate the Fisher discriminant classifier
# ----------------------------------------------------------------------------------------------------
fisher = Fisher((data_train, labels_train), n_components=1)#, class_weight='class_mass')
fisher.fit()
# log.info(f"Trained Fisher LDA classifier: w = {fisher.w.T}")
fisher_preds = fisher.predict(data_test.copy())
cm, F1_fisher = accuracy(labels_test, fisher_preds)
print(f"Fisher_F1 = {F1_fisher}")
print(cm)
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Instantiate the Logistic classifier
# ----------------------------------------------------------------------------------------------------
labels_train_logistic = labels_train.copy()
labels_test_logistic = labels_test.copy()
labels_train_logistic[labels_train_logistic < 0] = 0
labels_test_logistic[labels_test_logistic < 0] = 0
logistic = Logistic((data_train, labels_train_logistic), learning_rate=0.1, n_iters=data.shape[0], reg=0*0.015, order=3, method="logistic")#, class_weight='class_mass')
logistic.fit_transform()
# log.info(f"Trained logistic classifier (order 5): w = {logistic.w.T}")
logistic_preds = logistic.predict(data_test.copy())
cm, F1_logistic = accuracy(labels_test_logistic, logistic_preds)
print(f"Logistic_order_5_F1 = {F1_logistic}")
print(cm)
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Instantiate the SKLearn SVM classifier
# --------------------------------------------------
svc = SVM((data_train, labels_train), order=1, c=1, gamma=0.001, kernel='rbf')#, class_weight='class_mass')
# svc.optimize_c_gamma()
svc.fit()
# log.info(f"Trained linear classifier (No regularization + uniform class weights): w = {svc.w.T}")
svc_preds = svc.predict(data_test.copy())
cm, F1_svc = accuracy(labels_test, svc_preds)
print(f"SVM_F1 = {F1_svc}")
print(cm)
# --------------------------------------------------