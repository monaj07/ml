#
import logging as log
from matplotlib import pyplot as plt
import numpy as np
from ml.clustering import Kmeans, MeanShift, SpectralClustering
from utils import accuracy

log.basicConfig(level=log.INFO)


### Generate synthetic data using Gaussians
sizes = [100, 100, 100]
# class0:
mu0 = np.array([0, 10])
cov0 = np.array([[2, 0], [0, 2]]) * 0.5
data0 = np.random.multivariate_normal(mu0, cov0, size=sizes[0])
# class1:
mu1 = np.array([0, 0])
cov1 = np.array([[500, 0], [0, 1]]) * 0.5
data1 = np.random.multivariate_normal(mu1, cov1, size=sizes[1])
# class2:
mu2 = np.array([0, -10])
cov2 = np.array([[2, 0], [0, 2]]) * 0.5
data2 = np.random.multivariate_normal(mu2, cov2, size=sizes[2])

### Combine data from different classes, shuffle them and split it into train and test sets
data = np.vstack([data0, data1, data2])
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
fig, ax = plt.subplots(1, 7)
class_colours = ['r', 'g', 'b'] 
idx_grid = [np.where(labels_train == c)[0] for c in classes]
for c in classes:
    ax[0].plot(data_train[idx_grid[c], 0], data_train[idx_grid[c], 1], '.', color=class_colours[c])

ax[0].set_title("Real data")
# ------------------------------------------------------------------------------------------
### Kmeans
# ---------------------------------------------
kmeans = Kmeans(data=(data_train, labels_train), k=3, order=1, n_epochs=10, method="kmeans")
kmeans.fit()
predictions_kmeans = kmeans.idx
# ---------------------------------------------
idx_grid = [np.where(predictions_kmeans == c)[0] for c in classes]
for c in classes:
    ax[1].plot(data_train[idx_grid[c], 0], data_train[idx_grid[c], 1], '.', color=class_colours[c])
ax[1].set_title("kmeans")
# ---------------------------------------------
### Kmeans sklearn:
# ---------------------------------------------
kmeans_sk = Kmeans(data=(data_train, labels_train), k=3, order=1, n_epochs=10, method="kmeans_sklearn")
kmeans_sk.fit()
predictions_kmeans_sk = kmeans_sk.idx
# ---------------------------------------------
idx_grid = [np.where(predictions_kmeans_sk == c)[0] for c in classes]
for c in classes:
    ax[2].plot(data_train[idx_grid[c], 0], data_train[idx_grid[c], 1], '.', color=class_colours[c])

ax[2].set_title("kmeans_sklearn")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
### MeanShift
# ---------------------------------------------
meanshift = MeanShift(data=(data_train, labels_train), radius=6, order=1, n_epochs=1000, method="meanshift")
meanshift.fit()
predictions_meanshift = meanshift.idx
# ---------------------------------------------
classes = np.unique(predictions_meanshift)
idx = [np.where(predictions_meanshift == c)[0] for c in classes]
for c in classes:
    ax[3].plot(data_train[idx[c], 0], data_train[idx[c], 1], '.')
ax[3].set_title(f"meanshift\n{classes.size} clusters")
# ---------------------------------------------
### Meanshift sklearn:
# ---------------------------------------------
meanshift_sklearn = MeanShift(data=(data_train, labels_train), radius=5, method="meanshift_sklearn")
meanshift_sklearn.fit()
predictions_meanshift_sklearn = meanshift_sklearn.idx
# ---------------------------------------------
classes = np.unique(predictions_meanshift_sklearn)
idx_grid = [np.where(predictions_meanshift_sklearn == c)[0] for c in classes]
for c in classes:
    ax[4].plot(data_train[idx_grid[c], 0], data_train[idx_grid[c], 1], '.')

ax[4].set_title(f"meanshift_sklearn\n{classes.size} clusters")
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
### SpectralClustering
# ---------------------------------------------
# meanshift = MeanShift(data=(data_train, labels_train), radius=6, order=1, n_epochs=1000, method="meanshift")
# meanshift.fit()
# predictions_meanshift = meanshift.idx
# # ---------------------------------------------
# classes = np.unique(predictions_meanshift)
# idx = [np.where(predictions_meanshift == c)[0] for c in classes]
# for c in classes:
#     ax[3].plot(data_train[idx[c], 0], data_train[idx[c], 1], '.')
# ax[3].set_title(f"meanshift\n{classes.size} clusters")
# ---------------------------------------------
### SpectralClustering sklearn:
# ---------------------------------------------
# data_train = data_train[labels_train!=2]
spectral_sklearn = SpectralClustering(data=(data_train, labels_train), k=3, method="spectral_sklearn")
spectral_sklearn.fit_transform()
predictions_spectral_sklearn = spectral_sklearn.idx
# ---------------------------------------------
classes = np.unique(predictions_spectral_sklearn)
idx_grid = [np.where(predictions_spectral_sklearn == c)[0] for c in classes]
for c in classes:
    ax[6].plot(data_train[idx_grid[c], 0], data_train[idx_grid[c], 1], '.')

ax[6].set_title(f"spectral_sklearn")
# ------------------------------------------------------------------------------------------

plt.show()