#
from abc import ABC, abstractmethod
import numpy as np
from sklearn import linear_model
from .base import Algorithm


class GLR(Algorithm):
    """
    Generalized Linear Regression:
    It applies non-linear polynomial transformations of orders >= 1 to the features.
    """
    def __init__(self, data=None, reg=0, order=1, method="analytic"):
        super().__init__(method=method, order=order, data=data)
        self.w = None
        self.reg = reg
        self.model = None

    def fit_analytic(self):
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        xTx = self.x.T.dot(self.x)
        xTy = self.x.T.dot(self.y)
        self.w = np.linalg.inv(xTx + self.reg * np.eye(xTx.shape[0])).dot(xTy)
    
    def fit_sklearn(self):
        self.model = linear_model.LinearRegression()
        self.model.fit(self.x, self.y)

    def predict(self, data=None):
        super().predict(data)
        if self.method == "sklearn":
            return self.predictions
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        return self.x.dot(self.w)


class GLR_SGD(GLR):
    """
    Generalized Linear Regression:
    It applies non-linear polynomial transformations of orders >= 1 to the features.
    """

    def __init__(self, data=None, reg=0, order=1, learning_rate=0.05, n_iters=3000):
        super().__init__(data, reg, method="sgd", order=order)
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit_sgd(self, learning_rate=None, n_iters=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_iters is None:
            n_iters = self.n_iters
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        self.w = np.random.randn(self.x.shape[1], 1)
        self.w[-1] = 0  # bias
        for ii in range(n_iters):
            i = ii % self.x.shape[0]
            error = (self.y[i] - self.x[i, :].reshape(self.w.shape).T.dot(self.w))
            grad = -self.x[i, :].reshape(self.w.shape) * error + self.reg * self.w
            self.w += -learning_rate * grad
            # print(f"{ii}: {error}")
            # print(self.w.T)


class GLR_RANSAC(GLR):
    """
    Generalized Linear Regression using RANSAC method:
    """
    def __init__(self, data=None, order=1, n_iters=3000, inlier_margin=1):
        super().__init__(method="ransac", order=order, data=data)
        self.w = None
        self.n_iters = n_iters
        self.inlier_margin = inlier_margin

    def fit_ransac(self, n_iters=None):
        if n_iters is None:
            n_iters = self.n_iters
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        k_max = 0
        n_samples = self.x.shape[1]
        n_data = self.x.shape[0]
        for ii in range(n_iters):
            indices = np.random.choice(n_data, n_samples, replace=False)
            x_samples = self.x[indices, :]
            y_samples = self.y[indices]
            w = np.linalg.solve(x_samples, y_samples)
            y_hat = self.x.dot(w)
            n_inliers = sum(np.abs(self.y - y_hat) < self.inlier_margin)
            if k_max < n_inliers:
                k_max = n_inliers
                self.w = w
                print(f"iteration: {ii}/{n_iters} - n_inlires: {k_max} - w: {w.T}")