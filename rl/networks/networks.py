"""
Define the Deep Neural Networks that are used in DRL algorithms,
in this file.
"""
import torch
import torch.nn as nn


class ActorDDPG(nn.Module):
    def __init__(self, input_size, action_dimension):
        super().__init__()
        self.input_size = input_size
        self.action_dimension = action_dimension
        if isinstance(input_size, tuple):
            h, w = input_size
            conv_layers = list()
            conv_layers.append(nn.Conv2d(3, 16, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(16))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Conv2d(16, 32, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(32))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Conv2d(32, 32, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(32))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Flatten())

            self.conv_layers = nn.Sequential(*conv_layers)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=5, stride=2):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            input_size = convw * convh * 32

        dense_layers = list()
        dense_layers.append(nn.Linear(input_size, 20))
        dense_layers.append(nn.BatchNorm1d(20))
        dense_layers.append(nn.ReLU6())
        dense_layers.append(nn.Linear(20, action_dimension))
        # dense_layers.append(nn.Tanh())

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x):
        if isinstance(self.input_size, tuple):
            x = self.conv_layers(x)
        actions = self.dense_layers(x)
        return actions


class CriticDDPG(nn.Module):
    def __init__(self, input_size, action_dimension):
        super().__init__()
        self.input_size = input_size
        if isinstance(input_size, tuple):
            h, w = input_size
            conv_layers = list()
            conv_layers.append(nn.Conv2d(3, 16, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(16))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Conv2d(16, 32, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(32))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Conv2d(32, 32, kernel_size=5, stride=2))
            conv_layers.append(nn.BatchNorm2d(32))
            conv_layers.append(nn.ReLU6())
            conv_layers.append(nn.Flatten())

            self.conv_layers = nn.Sequential(*conv_layers)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=5, stride=2):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            input_size = convw * convh * 32

        dense_base_layers = list()
        dense_base_layers.append(nn.Linear(input_size, 20))
        dense_base_layers.append(nn.BatchNorm1d(20))
        dense_base_layers.append(nn.ReLU())

        self.dense_base_layers = nn.Sequential(*dense_base_layers)

        dense_layers = list()
        dense_layers.append(nn.Linear(20 + action_dimension, 20))
        dense_layers.append(nn.BatchNorm1d(20))
        dense_layers.append(nn.ReLU6())
        dense_layers.append(nn.Linear(20, 1))
        # dense_layers.append(nn.BatchNorm1d(20))
        # dense_layers.append(nn.ReLU6())
        # dense_layers.append(nn.Linear(20, 1))

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x, action_input):
        if isinstance(self.input_size, tuple):
            x = self.conv_layers(x)
        dense_base_output = self.dense_base_layers(x)
        dense_rest_input = torch.cat([dense_base_output, action_input], dim=1)
        q_value = self.dense_layers(dense_rest_input)
        return q_value
