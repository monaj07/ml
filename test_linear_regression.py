#
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import logging as log
from ml import *
from ml.ensemble import GBRegression
from sklearn.metrics import mean_squared_error

log.basicConfig(level=log.INFO)

# ############# MLR
# # Synthetic data generation for an MLR
# N = 500  # number of samples
# std = 5
# eps = np.random.randn(N) * std
# a, b = np.array([4, 8]), 3
# x_range = np.array([10, 5])
# x = np.random.rand(N, a.size) * x_range
# y = x.dot(a) + b + eps
# lr_data = (x, y)

# # Fitting the simple LR model to the data
# lr = LR(data=lr_data, reg=1)
# lr.fit()
# w = lr.w
# log.info(f"Estimated parameters: {w}")
# predictions = lr.predict()

# ax = plt.figure().gca(projection='3d')
# ax.scatter(x[:, 0], x[:, 1], y, '*')
# X0 = np.arange(x[:, 0].min(), x[:, 0].max(), 0.1)
# X1 = np.arange(x[:, 1].min(), x[:, 1].max(), 0.1)
# X0, X1 = np.meshgrid(X0, X1)
# Z = w[0] * X0 + w[1] * X1 + w[2]
# ax.plot_surface(X0, X1, Z)
# plt.show()

############# Generalized LR *****************************

### Synthetic data generation for Sin(x)
N = 100  # number of samples
Ts = int(0.6 * N)  # Training data size
std = 0.5
eps = np.random.randn(N) * std
x_range = 3*np.pi
x = (np.random.rand(N)-0.5) * x_range
# y = -x**4 + 3*x**3 + 5*x + 1 + eps
y = 5*np.cos(x) + 1 + eps
lr_data = (x[:Ts], y[:Ts])
lr_data_test = (x[Ts:], y[Ts:])
#----------------------------------------------------------------

### Simple Linear Regression
lr = GLR(data=lr_data, reg=0, order=1, method='analytic')
lr.fit()
w = lr.w
# log.info(f"Estimated parameters: {w.T}")
predictions_lr = lr.predict(lr_data_test[0])
mse_lr = mean_squared_error(lr_data_test[1], predictions_lr)
#----------------------------------------------------------------

### Polynomial non-linear features, then linear regression (overfitted to order 15!, but regularized)
glr_overfit_but_regularized = GLR(data=lr_data, reg=50, order=15, method='analytic')
glr_overfit_but_regularized.fit()
w_glr_overfit_but_regularized = glr_overfit_but_regularized.w
# log.info(f"Estimated glr_overfit_but_regularized parameters: {w_glr_overfit_but_regularized.T}")
predictions_glr_overfit_but_regularized = glr_overfit_but_regularized.predict(lr_data_test[0])
mse_glr_overfit = mean_squared_error(lr_data_test[1], predictions_glr_overfit_but_regularized)
# log.info(f"sum_w_glr_overfit_but_regularized = {(w_glr_overfit_but_regularized**2).sum()}")
#----------------------------------------------------------------

### Polynomial non-linear features, then linear regression
glr = GLR(data=lr_data, reg=1, order=4, method='analytic')
glr.fit()
w_glr2 = glr.w
# log.info(f"Estimated glr2 parameters: {w_glr2.T}")
predictions_glr2 = glr.predict(lr_data_test[0])
mse_glr2 = mean_squared_error(lr_data_test[1], predictions_glr2)
# log.info(f"sum_w_glr2 = {(w_glr2**2).sum()}")
#----------------------------------------------------------------

### Polynomial non-linear features (sklearn), then Linear Regression of sklearn
glr_sk = GLR(data=lr_data, reg=1, order=4, method="sklearn")
glr_sk.fit()
w_glr_sk = glr_sk.w
predictions_glr_sk = glr_sk.predict(lr_data_test[0])
mse_glr_sk = mean_squared_error(lr_data_test[1], predictions_glr_sk)
#----------------------------------------------------------------

### GLR_SGD
glr_sgd_new = GLR_SGD(data=lr_data, reg=0.00, order=4, learning_rate=0.05, n_iters=3000)
glr_sgd_new.fit_transform()
w_glr_sgd_new = glr_sgd_new.w
# log.info(f"Estimated w_glr_sgd_new parameters: {w_glr_sgd_new.T}")
predictions_glr_sgd_new = glr_sgd_new.predict(lr_data_test[0])
mse_glr_sgd_new = mean_squared_error(lr_data_test[1], predictions_glr_sgd_new)
#----------------------------------------------------------------

### GLR_RANSAC
glr_ransac = GLR_RANSAC(data=lr_data, order=1, n_iters=500, inlier_margin=1)
glr_ransac.fit()
w_glr_ransac = glr_ransac.w
# log.info(f"Estimated w_glr_ransac parameters: {w_glr_ransac.T}")
predictions_glr_ransac = glr_ransac.predict(lr_data_test[0])
mse_glr_ransac = mean_squared_error(lr_data_test[1], predictions_glr_ransac)
#----------------------------------------------------------------

### Gradient Boosted Regression
gbr = GBRegression(data=lr_data, order=1, n_estimators=500, max_depth=4)
gbr.fit()
predictions_gbr = gbr.predict(lr_data_test[0])
mse_gbr = mean_squared_error(lr_data_test[1], predictions_gbr)
#----------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(x[Ts:].reshape(-1, 1), y[Ts:], '*', label='Real data')
ax.plot(x[Ts:].reshape(-1, 1), predictions_lr, '.r', label=f'Simple linear regression, mse={mse_lr}')
# # ax.plot(x[Ts:].reshape(-1, 1), predictions_glr_overfit_but_regularized, 'og', label='Overfit (high order) but regularized regression')
ax.plot(x[Ts:].reshape(-1, 1), predictions_glr2, '.b', label=f'Regression with right polynomial, mse={mse_glr2}')
ax.plot(x[Ts:].reshape(-1, 1), predictions_glr_sk, '.k', label=f'Regression with sklearn polynomial, mse={mse_glr_overfit}')
ax.plot(x[Ts:].reshape(-1, 1), predictions_glr_sgd_new, '.y', label=f'Regression with GLR_SGD with polynomial, mse={mse_glr_sgd_new}')
ax.plot(x[Ts:].reshape(-1, 1), predictions_glr_ransac, '.g', label=f'Regression with GLR_RANSAC with polynomial, mse={mse_glr_ransac}')
ax.plot(x[Ts:].reshape(-1, 1), predictions_gbr, '.m', label=f'Regression with GBR with polynomial, mse={mse_gbr}')
ax.legend()
plt.show()