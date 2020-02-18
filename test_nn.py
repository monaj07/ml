#
import logging as log
from matplotlib import pyplot as plt
import numpy as np
from ml.classification import SoftmaxRegression
from ml.nn import NN
from ml.ensemble import GBClassification, RandomForest
from utils import accuracy

log.basicConfig(level=log.INFO)


### Generate synthetic data using Gaussians
sizes = [100, 100]
# class0:
mu0 = np.array([-3, 5])
cov0 = np.array([[13, 11], [11, 13]]) * 0.5
data0 = np.random.multivariate_normal(mu0, cov0, size=sizes[0]//2)
mu00 = np.array([5, 5])
cov00 = np.array([[13, -11], [-11, 13]]) * 0.5
data00 = np.random.multivariate_normal(mu00, cov00, size=sizes[0]//2)
data0 = np.vstack([data0, data00])
# class1:
mu1 = np.array([0, 5])
cov1 = np.array([[13, -11], [-11, 13]]) * 0.5
data1 = np.random.multivariate_normal(mu1, cov1, size=sizes[1])

### Combine data from different classes, shuffle them and split it into train and test sets
data = np.vstack([data0, data1])
labels = np.concatenate([i * np.ones(sizes[i]) for i in range(len(sizes))]).astype(int)
N = sum(sizes)
Ts = int(0.6 * N)
rand_idx = np.random.permutation(N)
data = data[rand_idx]
labels = labels[rand_idx]
data_train = data[:Ts, :]
labels_train = labels[:Ts]
data_test = data[Ts:, :]
labels_test = labels[Ts:]
classes = np.unique(labels)

idx_train = [np.where(labels_train == c)[0] for c in classes]
idx_test = [np.where(labels_test == c)[0] for c in classes]

###############################################
### Plot data:
fig, ax = plt.subplots(1, 4)
class_colours = ['r', 'g', 'b']

# ------------------------------------------------------------------------------------------
### Softmax Regression
# ---------------------------------------------
softmax = SoftmaxRegression(data=(data_train, labels_train), learning_rate=0.01, reg=0.05*0, order=1, n_iters=1000)
softmax.fit_transform()
predictions_softmax = softmax.predict(data_test)
cm, F1 = accuracy(labels_test, predictions_softmax)
log.info(f"F1_softmax = {F1}")
log.info(f"weight = {softmax.w}")
print(cm)
# ---------------------------------------------
# Classify all grid points to visualize decision boundaries
data_grid = np.array([(x0, x1) for x0 in np.arange(data_test[:, 0].min(), data_test[:, 0].max(), .1)
                               for x1 in np.arange(data_test[:, 1].min(), data_test[:, 1].max(), .1)])
data_grid_norm = data_grid if softmax.mean is None else (data_grid - softmax.mean) / softmax.std
predictions_data_grid_softmax = softmax.predict(data_grid)
idx_grid = [np.where(predictions_data_grid_softmax == c)[0] for c in classes]
for c in classes:
    ax[0].plot(data_grid_norm[idx_grid[c], 0], data_grid_norm[idx_grid[c], 1], '.', color=class_colours[c])
# Visualize our own data points
data_test_norm = data_test if softmax.mean is None else (data_test - softmax.mean) / softmax.std
for c in classes:
    ax[0].plot(data_test_norm[idx_test[c], 0], data_test_norm[idx_test[c], 1], 'o', markeredgecolor='k', color=class_colours[c])

# # ---------------------------------------------
# ### This is how the decision boundaries of multi-class problem are computed
# ww01 = softmax.w[:, 0] - softmax.w[:, 1]
# ww02 = softmax.w[:, 0] - softmax.w[:, 2]
# ww12 = softmax.w[:, 1] - softmax.w[:, 2]
# x_range = np.linspace(data_test_norm[predictions_softmax==0, 0].min(), data_test_norm[predictions_softmax==0, 0].max(), 100)
# ax[0].plot(x_range, -ww01[0]/ww01[1] * x_range - ww01[2]/ww01[1], '-y')
# x_range = np.linspace(data_test_norm[predictions_softmax==1, 0].min(), data_test_norm[predictions_softmax==1, 0].max(), 100)
# ax[0].plot(x_range, -ww02[0]/ww02[1] * x_range - ww02[2]/ww02[1], '-y')
# x_range = np.linspace(data_test_norm[predictions_softmax==2, 0].min(), data_test_norm[predictions_softmax==2, 0].max(), 100)
# ax[0].plot(x_range, -ww12[0]/ww12[1] * x_range - ww12[2]/ww12[1], '-y')
# # ---------------------------------------------

ax[0].set_title("Softmax Regression")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
### Softmax Regression from SKLEARN
# ---------------------------------------------
softmax_sklearn = SoftmaxRegression(data=(data_train, labels_train), order=1, method="softmax_sklearn")
softmax_sklearn.fit_transform()
predictions_softmax_sklearn = softmax_sklearn.predict(data_test)
cm, F1 = accuracy(labels_test, predictions_softmax_sklearn)
log.info(f"F1_softmax_sklearn = {F1}")
# log.info(f"weight = {softmax_sklearn.w}")
print(cm)

# ------------------------------------------------------------------------------------------
### Gradient Boosted Classification
# ---------------------------------------------
gbc = GBClassification(data=(data_train, labels_train))
gbc.fit()
predictions_gbc = gbc.predict(data_test)
cm, F1 = accuracy(labels_test, predictions_gbc)
log.info(f"F1_GradientBoostedClassigication = {F1}")
print(cm)
# ---------------------------------------------
# Classify all grid points to visualize decision boundaries
data_grid = np.array([(x0, x1) for x0 in np.arange(data_test[:, 0].min(), data_test[:, 0].max(), .1)
                               for x1 in np.arange(data_test[:, 1].min(), data_test[:, 1].max(), .1)])
data_grid_norm = data_grid if gbc.mean is None else (data_grid - gbc.mean) / gbc.std
predictions_data_grid_gbc = gbc.predict(data_grid)
idx_grid = [np.where(predictions_data_grid_gbc == c)[0] for c in classes]
for c in classes:
    ax[1].plot(data_grid_norm[idx_grid[c], 0], data_grid_norm[idx_grid[c], 1], '.', color=class_colours[c])
# Visualize our own data points
data_test_norm = data_test if gbc.mean is None else (data_test - gbc.mean) / gbc.std
for c in classes:
    ax[1].plot(data_test_norm[idx_test[c], 0], data_test_norm[idx_test[c], 1], 'o', markeredgecolor='k', color=class_colours[c])
ax[1].set_title("Gradient Boosted Classification")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
### Random Forest Classification
# ---------------------------------------------
rf = RandomForest(data=(data_train, labels_train))
rf.fit()
predictions_rf = rf.predict(data_test)
cm, F1 = accuracy(labels_test, predictions_rf)
log.info(f"F1_RandomForest = {F1}")
print(cm)
# ---------------------------------------------
# Classify all grid points to visualize decision boundaries
data_grid = np.array([(x0, x1) for x0 in np.arange(data_test[:, 0].min(), data_test[:, 0].max(), .1)
                               for x1 in np.arange(data_test[:, 1].min(), data_test[:, 1].max(), .1)])
data_grid_norm = data_grid if rf.mean is None else (data_grid - rf.mean) / rf.std
predictions_data_grid_rf = rf.predict(data_grid)
idx_grid = [np.where(predictions_data_grid_rf == c)[0] for c in classes]
for c in classes:
    ax[2].plot(data_grid_norm[idx_grid[c], 0], data_grid_norm[idx_grid[c], 1], '.', color=class_colours[c])
# Visualize our own data points
data_test_norm = data_test if rf.mean is None else (data_test - rf.mean) / rf.std
for c in classes:
    ax[2].plot(data_test_norm[idx_test[c], 0], data_test_norm[idx_test[c], 1], 'o', markeredgecolor='k', color=class_colours[c])
ax[2].set_title("Random Forest")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
### NN
# ---------------------------------------------
nn = NN(data=(data_train, labels_train), learning_rate=0.01, wdecay=0.001, n_iters=1000)
nn.add_layer(out_size=10, activation="relu")
nn.add_layer(out_size=labels.max()+1, activation="linear")
nn.fit_transform()
predictions_nn = nn.predict(data_test)
cm, F1 = accuracy(labels_test, predictions_nn)
log.info(f"F1_nn = {F1}")
print(cm)
# ---------------------------------------------
# Classify all grid points to visualize decision boundaries
data_grid = np.array([(x0, x1) for x0 in np.arange(data_test[:, 0].min(), data_test[:, 0].max(), .1)
                               for x1 in np.arange(data_test[:, 1].min(), data_test[:, 1].max(), .1)])
data_grid_norm = data_grid if nn.mean is None else (data_grid - nn.mean) / nn.std
predictions_nn_data_grid = nn.predict(data_grid)
idx_grid = [np.where(predictions_nn_data_grid == c)[0] for c in classes]
for c in classes:
    ax[3].plot(data_grid_norm[idx_grid[c], 0], data_grid_norm[idx_grid[c], 1], '.', color=class_colours[c])
# Visualize our own data points
data_test_norm = data_test if nn.mean is None else (data_test - nn.mean) / nn.std
for c in classes:
    ax[3].plot(data_test_norm[idx_test[c], 0], data_test_norm[idx_test[c], 1], 'o', markeredgecolor='k', color=class_colours[c])
ax[3].set_title("Neural Network")
# ------------------------------------------------------------------------------------------

plt.show()