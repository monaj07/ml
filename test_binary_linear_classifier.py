#
import numpy as np
import matplotlib.pyplot as plt
from ml.classification import Perceptron, Fisher, LogisticRegression as Logistic, SVM
import logging as log
from mpl_toolkits.mplot3d import Axes3D
from utils import accuracy

log.basicConfig(level=log.INFO)


### Generate synthetic data using Gaussians
# class1:
size1 = 40
mean1 = np.array([-4, 3])
cov1 = np.array([[9, -3], [-3, 8]])
data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=size1)
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
log.info(f"Trained linear classifier (No regularization + uniform class weights): w = {percept.w.T}")
percept_preds = percept.predict(data_test.copy())
cm, F1_percept = accuracy(labels_test, percept_preds)
print(f"F1 = {F1_percept}")
print(cm)
# --------------------------------------------------
# --------------------------------------------------
fig, ax = plt.subplots(1, 3)
idx_c1 = labels_test < 0
idx_c2 = labels_test > 0
data_test_norm = data_test if percept.mean is None else (data_test - percept.mean) / percept.std 
ax[0].plot(data_test_norm[idx_c2, 0], data_test_norm[idx_c2, 1], 'ob')
ax[0].plot(data_test_norm[idx_c1, 0], data_test_norm[idx_c1, 1], 'or')
idx_wrong = np.where(percept_preds != labels_test.reshape(-1, 1))[0]
ax[0].plot(data_test_norm[idx_wrong, 0], data_test_norm[idx_wrong, 1], 'xk')
x_range = np.linspace(data_test_norm[:, 0].min(), data_test_norm[:, 0].max(), 100)
ax[0].plot(x_range, -percept.w[0]/percept.w[1] * x_range - percept.w[2]/percept.w[1], '-g')
ax[0].set_title("Perceptron")
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
log.info(f"Trained Fisher LDA classifier: w = {fisher.w.T}")
fisher_preds = fisher.predict(data_test.copy())
cm, F1_fisher = accuracy(labels_test, fisher_preds)
print(f"F1 = {F1_fisher}")
print(cm)
# --------------------------------------------------
# --------------------------------------------------
# data_test_norm = data_test - percept.mean
# data_test_norm /= percept.std
ax[1].plot(data_test[idx_c2, 0], data_test[idx_c2, 1], 'ob')
ax[1].plot(data_test[idx_c1, 0], data_test[idx_c1, 1], 'or')
idx_wrong = np.where(fisher_preds != labels_test)[0]
ax[1].plot(data_test[idx_wrong, 0], data_test[idx_wrong, 1], 'xk')
x_range = np.linspace(data_test[:, 0].min(), data_test[:, 0].max(), 100)
ax[1].plot(x_range, -fisher.w[0]/fisher.w[1] * x_range - fisher.w[2]/fisher.w[1], '-g')
ax[1].set_title("Fisher LDA")
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Instantiate the Logistic classifier
# ----------------------------------------------------------------------------------------------------
labels_train_logistic = labels_train.copy()
labels_test_logistic = labels_test.copy()
labels_train_logistic[labels_train_logistic < 0] = 0
labels_test_logistic[labels_test_logistic < 0] = 0
logistic = Logistic((data_train, labels_train_logistic), learning_rate=0.1, n_iters=data.shape[0], reg=0*0.015, method="logistic_sklearn")#, class_weight='class_mass')
logistic.fit()
log.info(f"Trained logistic classifier: w = {logistic.w.T}")
logistic_preds = logistic.predict(data_test.copy())
cm, F1_logistic = accuracy(labels_test_logistic, logistic_preds)
print(f"F1 = {F1_logistic}")
print(cm)
# --------------------------------------------------
# --------------------------------------------------
data_test_norm = data_test if logistic.mean is None else (data_test - logistic.mean) / logistic.std 
ax[2].plot(data_test_norm[idx_c2, 0], data_test_norm[idx_c2, 1], 'ob')
ax[2].plot(data_test_norm[idx_c1, 0], data_test_norm[idx_c1, 1], 'or')
idx_wrong = np.where(logistic_preds.squeeze() != labels_test_logistic)[0]
ax[2].plot(data_test_norm[idx_wrong, 0], data_test_norm[idx_wrong, 1], 'xk')
x_range = np.linspace(data_test_norm[:, 0].min(), data_test_norm[:, 0].max(), 100)
ax[2].plot(x_range, -logistic.w[0]/logistic.w[1] * x_range - logistic.w[2]/logistic.w[1], '-g')
ax[2].set_title("Logistic Regression")
# ----------------------------------------------------------------------------------------------------
plt.show()
