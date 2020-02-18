#
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Algorithm(ABC):
    @abstractmethod
    def __init__(self, data=None, method=None, order=None):
        self.method = method
        self.order = order
        self.mean = None
        self.std = None
        if data is not None:
            self.y = data[1].copy()
            self.x = data[0].copy()

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, _y):
        if _y.ndim == 1:
            _y = _y[:, np.newaxis]
        # assert (len(set(np.unique(_y)) - {-1, 1}) == 0), "The labels for the binary perceptron should be either -1 or 1"
        self._y = _y

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, _x):
        if _x.ndim == 1:
            _x = _x[:, np.newaxis]
        self._x = _x

    # @abstractmethod
    def fit(self):
        self._polynomial_features()
        getattr(self, f"fit_{self.method}")()

    def fit_transform(self):
        self._polynomial_features()
        self.mean = self.x.mean(0)
        self.std = self.x.std(0)
        self.x = (self.x - self.mean) / self.std
        getattr(self, f"fit_{self.method}")()
    
    def _polynomial_features(self):
        self.poly = PolynomialFeatures(degree=self.order)
        self.x = self.poly.fit_transform(self.x)[:, 1:]
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]
        # if self.method == "sklearn":
        #     self.poly = PolynomialFeatures(degree=self.order)
        #     self.x = self.poly.fit_transform(self.x)
        # else:
        #     self.x = np.hstack([(self.x)**i for i in range(1, self.order + 1)])

    def predict(self, data=None):
        self.x = data
        if self.method == "sklearn":
            self.x = self.poly.fit_transform(self.x)[:, 1:]
            self.predictions = self.model.predict(self.x)
        self._polynomial_features()
        if self.mean is not None:
            self.x = (self.x - self.mean) / self.std