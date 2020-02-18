#
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import logging as log
from ml import GLR, GLR_RANSAC

log.basicConfig(level=log.INFO)

################################################################
################ Test RANSAC
################################################################

np.random.seed(50)
### Synthetic data generation for Sin(x)
N = 100  # number of samples
Ts = int(0.6 * N)  # Training data size
std = 5
eps = np.random.randn(N) * std
x_range = 10
x = np.random.rand(N) * x_range
y = 5*x + 9 + eps
# Add random outliers to the dataset:
outliers_indices = np.random.choice(N, N//10, replace=False)
y[outliers_indices] += 10 * np.abs(eps[outliers_indices]) * x[outliers_indices]
#
lr_data = (x[:Ts], y[:Ts])
lr_data_test = (x[Ts:], y[Ts:])
#----------------------------------------------------------------


### Simple Linear Regression
lr = GLR(data=lr_data, reg=0, order=1, method='analytic')
lr.fit()
w = lr.w
log.info(f"Estimated parameters: {w.T}")
predictions_lr = lr.predict(lr_data_test[0])
#----------------------------------------------------------------


### GLR_RANSAC
glr_ransac = GLR_RANSAC(data=lr_data, order=1, n_iters=500, inlier_margin=1)
glr_ransac.fit()
w_glr_ransac = glr_ransac.w
log.info(f"Estimated w_glr_ransac parameters: {w_glr_ransac.T}")
predictions_glr_ransac = glr_ransac.predict(lr_data_test[0])
#----------------------------------------------------------------


### Simple Linear Regression with heavy regularization to possibly diminish the impact of outliers
lr_reg = GLR(data=lr_data, reg=2000, order=1, method='analytic')
lr_reg.fit()
w_lr_reg = lr_reg.w
log.info(f"Estimated parameters: {w_lr_reg.T}")
predictions_lr_reg = lr_reg.predict(lr_data_test[0])
#----------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(x[Ts:].reshape(-1, 1), y[Ts:], '*', label='Real data')
ax.plot(x[Ts:].reshape(-1, 1), predictions_lr, '.r', label='Simple LR')
ax.plot(x[Ts:].reshape(-1, 1), predictions_glr_ransac, '.k', label='LR with GLR_RANSAC')
ax.plot(x[Ts:].reshape(-1, 1), predictions_lr_reg, '.y', label='Simple LR with heavy regularization')
ax.legend()
plt.show()