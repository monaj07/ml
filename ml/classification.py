#
from .base import Algorithm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKLearn


class Perceptron(Algorithm):
    def __init__(self, data, learning_rate=0.05, n_iters=1000, reg=0.1, class_weight='class_mass'):
        super().__init__(data=data, order=1, method="percept")
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.reg = reg
        self.class_weight = class_weight

    # def def_getter_setter(func):
    #     func_name = '_' + func.__name__
    #     @property
    #     def func(self):
    #         return getattr(self, func_name)
    #     @func.setter
    #     def func(self, val):
    #         if val.ndim == 1:
    #             val = val[:, np.newaxis]
    #         setattr(self, func_name, val)
    #     return func

    # @def_getter_setter
    # def y(self):
    #     pass

    # @def_getter_setter
    # def x(self):
    #     pass

    def fit_percept(self, learning_rate=None, n_iters=None, class_weight=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_iters is None:
            n_iters = self.n_iters
        if class_weight is None:
            class_weight = self.class_weight

        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        self.w = np.random.randn(self.x.shape[1], 1)

        if class_weight == 'uniform':
            cw = {-1: 0.5, 1: 0.5}
        elif class_weight == 'class_mass':
            cw = {}
            cw[-1] = np.sum(self.y == -1) / self.y.size
            cw[1] = np.sum(self.y == 1) / self.y.size
        elif isinstance(class_weight, (list, tuple, np.ndarray)):
            cw = {-1: class_weight[0], 1: class_weight}
        else:
            raise NotImplementedError
        for ii in range(n_iters):
            i = ii % self.x.shape[0]
            if self.y[i] * self.x[i, :].T.dot(self.w) > 0:
                # The prediction is correct, so move on to another data point
                continue
            grad = -self.y[i] * self.x[i, :].T.reshape(-1, 1) * (1 - cw[self.y[i].item()]) + self.reg * self.w
            self.w -= learning_rate * grad

    def predict(self, data):
        super().predict(data)
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        predictions = np.sign(self.x.dot(self.w))
        return predictions


class Fisher(Algorithm):
    def __init__(self, data=None, order=1, n_components=1):
        super().__init__(data=data, method="fisher", order=order)
        self.order = order

    def fit_fisher(self):
        self.model = LDA(n_components=1)
        self.model.fit(self.x, self.y.squeeze())
        self.w = np.hstack([self.model.coef_[0], self.model.intercept_[0]])

    def predict(self, data=None):
        super().predict(data)
        predictions = self.model.predict(self.x)
        return predictions


def sig(a):
    return 1 / (1 + np.exp(-a))


class LogisticRegression(Algorithm):
    def __init__(self, data=None, learning_rate=0.01, reg=0.1, order=1, n_iters=1000, method="logistic"):
        super().__init__(method=method, order=order)
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_iters = n_iters
        self.x = data[0]
        self.y = data[1]

    def define_getter_setter(func):
        func_name = func.__name__
        @property
        def func(self):
            return getattr(self, '_'+func_name)
        @func.setter
        def func(self, value):
            if value.ndim == 1:
                value = value[:, np.newaxis]
            setattr(self, '_'+func_name, value)
        return func

    @define_getter_setter
    def x(self):
        pass

    @define_getter_setter
    def y(self):
        pass

    def fit_logistic_sklearn(self):
        self.model = LogisticRegressionSKLearn(penalty='l2')
        self.model.fit(self.x, self.y.squeeze())
        self.w = np.hstack([self.model.coef_[0], self.model.intercept_[0]])

    def fit_logistic(self):
        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        w = np.random.randn(self.x.shape[1], 1)
        # fig, ax = plt.subplots()

        # losses = []
        for ii in range(self.n_iters):
            i = ii % self.x.shape[0]
            x, y = self.x[i, :].reshape(-1, 1), self.y[i]
            grad = x * (sig(x.T.dot(w)) - y)
            w -= self.learning_rate * grad + self.reg * w 
            # loss = -(y * np.log(sig(x.T.dot(w))) + (1 - y) * np.log(1 - sig(x.T.dot(w)))).squeeze()
            # losses.append(loss)
        # ax.plot(np.arange(self.n_iters), np.array(losses))
        # plt.show()
        self.w = w
    
    def predict(self, data=None):
        super().predict(data)
        if self.method == "logistic_sklearn":
            predictions = self.model.predict(self.x)
        else:
            self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
            probs = sig(self.x.dot(self.w))
            predictions = np.heaviside(probs - 0.5, 1)
        return predictions


class SVM(Algorithm):
    def __init__(self, data=None, order=1, c=1, gamma=0.1, kernel='linear'):
        super().__init__(method='svm', order=order, data=data)
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.model = SVC(gamma=self.gamma, C=self.c, kernel=kernel)
    
    def optimize_c_gamma(self):
        c_range = np.logspace(-2, 10, 13)
        g_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=g_range, C=c_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        grid = GridSearchCV(self.model, param_grid, cv=cv)
        grid.fit(self.x.squeeze(), self.y.squeeze())
        print(f"The best parameters are {grid.best_params_} with a score of {grid.best_score_}.")
        self.c = grid.best_params_['C']
        self.gamma = grid.best_params_['gamma']
    
    def fit_svm(self):
        self.model = SVC(gamma=self.gamma, C=self.c, kernel=self.kernel)
        self.model.fit(self.x.squeeze(), self.y.squeeze())

    def predict(self, data=None):
        super().predict(data)
        predictions = self.model.predict(self.x)
        return predictions


def softmax(u):
    return np.exp(-u) / np.exp(-u).sum()

class SoftmaxRegression(Algorithm):
    def __init__(self, data=None, learning_rate=0.01, reg=0.1, order=1, n_iters=1000, method="softmax"):
        super().__init__(data=data, method=method, order=order)
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_iters = n_iters
        self.x = data[0]
        self.y = data[1]

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, yin):
        """ One-hot encoding"""
        one_hot = np.zeros((yin.size, yin.max() + 1))
        one_hot[np.arange(yin.size), yin] = 1
        self._y = one_hot

    def fit_softmax(self, learning_rate=None, n_iters=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_iters is None:
            n_iters = self.n_iters

        self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
        w = np.random.randn(self.x.shape[1], self.y.shape[1])

        # fig, ax = plt.subplots()
        losses = []
        for ii in range(n_iters):
            i = ii % self.x.shape[0]
            x, y = self.x[i:i+1, :], self.y[i:i+1, :]
            p = softmax(x.dot(w))
            grad = -x.T.dot(p - y)
            w -= learning_rate * grad + self.reg * w
        #     loss = -y.dot(np.log(p).T).squeeze()
        #     losses.append(loss)
        # ax.plot(np.arange(self.n_iters), np.array(losses))
        # plt.show()
        self.w = w

    def fit_softmax_sklearn(self, learning_rate=None, n_iters=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_iters is None:
            n_iters = self.n_iters
        
        self.model = LogisticRegressionSKLearn(solver='lbfgs', multi_class='multinomial')
        self.model.fit(self.x, np.argmax(self.y, 1))


    def predict(self, data):
        super().predict(data)
        if self.method == "softmax_sklearn":
            predictions = self.model.predict(self.x)
        else:
            self.x = np.hstack([self.x, np.ones((self.x.shape[0], 1))])
            probs = softmax(self.x.dot(self.w))
            predictions = np.argmax(probs, 1)
        return predictions


