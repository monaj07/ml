#
from abc import ABC, abstractmethod
import numpy as np
from .base import Algorithm


class Activation(ABC):
    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def grad(self):
        pass


class Linear(Activation):
    def __call__(self, input):
        return input
    
    @staticmethod
    def grad(input):
        return np.ones_like(input)


class Relu(Activation):
    def __call__(self, input):
        return (input * (input > 0)).astype(input.dtype)
    
    @staticmethod
    def grad(input):
        return (input > 0).astype(input.dtype)


class Softmax(Activation):
    def __call__(self, input):
        return np.exp(input) / np.exp(input).sum()
    
    @staticmethod
    def grad(input):
        raise NotImplementedError


class Activations:
    def __init__(self):
        self.relu = Relu()
        self.linear = Linear()
        self.sigmoid = None
        self.tanh = None


class Optimize:
    def __init__(self, layers, learning_rate=0.01, wdecay=0.001):
        self._layers = layers
        self.lr = learning_rate
        self.wdecay = wdecay

    def step(self):
        for layer in self._layers:
            layer.w -= self.lr * layer.w_grad + self.wdecay * layer.w
            layer.b -= self.lr * layer.b_grad
        

class LossCESoftmax:
    def __init__(self, prediction, gt):
        self.loss = -(gt * np.log(prediction)).sum()
        self.grad = prediction - gt


class Layer:
    def __init__(self, in_size, out_size, activation="relu"):
        self.w = np.random.randn(out_size, in_size)
        self.b = np.zeros((out_size, 1))
        self.out_size = out_size
        self.activation = getattr(Activations(), activation)

    def __call__(self, input):
        self.input = input
        self.z = self.w.dot(input) + self.b
        return self.activation(self.z)

    def grad(self, error_in):
        self.w_grad = error_in * self.activation.grad(self.z) * self.input.T
        self.b_grad = error_in * self.activation.grad(self.z) * np.ones_like(self.b)
        self.error_out = self.w.T.dot(self.activation.grad(self.z) * error_in)
        return self.error_out


class NN(Algorithm):
    def __init__(self, data=None, learning_rate=0.01, wdecay=0.001, n_iters=1000, loss="cross_entropy", order=1, method="nn"):
        super().__init__(data=data, method=method, order=order)
        self.learning_rate = learning_rate
        self.wdecay = wdecay
        self.n_iters = n_iters
        self.loss = loss
        self.depth = 0
        self._layers = []

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _y):
        self.nclasses = _y.max() + 1
        self._y = np.zeros((_y.size, self.nclasses))
        self._y[np.arange(_y.size), _y] = 1

    def add_layer(self, out_size, activation="relu"):
        in_size = self.x.shape[1] if len(self._layers) == 0 else self._layers[-1].out_size
        self._layers.append(Layer(in_size, out_size, activation=activation))

    def forward(self, data=None):
        out = data
        for layer in self._layers:
            out = layer(out)
        self.predictions = Softmax()(out)
        return self.predictions
        
    def error(self, y):
        loss = LossCESoftmax(self.predictions, y)
        error_out = loss.grad
        return error_out
        
    def backward(self, y):
        error_in = self.error(y)
        for layer in self._layers[::-1]:
            error_out = layer.grad(error_in)
            error_in = error_out

    def fit_nn(self, n_iters=None):
        if n_iters is None:
            n_iters = self.n_iters
        optimizer = Optimize(self._layers, wdecay=self.wdecay)
        for ii in range(n_iters):
            i = ii % self.x.shape[0]
            x = self.x[i, :].reshape(-1, 1)
            y = self.y[i, :].reshape(-1, 1)
            self.forward(x)
            self.backward(y)
            optimizer.step()

    def predict(self, data=None):
        super().predict(data)
        predictions = []
        for i in range(self.x.shape[0]):
            out = self.x[i, :].reshape(-1, 1)
            for layer in self._layers:
                out = layer(out)
            predictions.append(Softmax()(out))
        predictions = np.array(predictions)
        self.predictions = np.argmax(predictions, 1).squeeze()
        return self.predictions