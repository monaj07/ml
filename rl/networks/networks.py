"""
Define the Deep Neural Networks that are used in DRL algorithms,
in this file.
"""
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, outputs, dueling_dqn=False):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.dueling_dqn = dueling_dqn

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        if self.dueling_dqn:
            self.value_stream = nn.Linear(linear_input_size, 1)
            self.advantage_stream = nn.Linear(linear_input_size, outputs)
        else:
            self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        if self.dueling_dqn:
            value_stream = self.value_stream(x.view(x.size(0), -1))
            advantage_stream = self.advantage_stream(x.view(x.size(0), -1))
            advantage_stream -= advantage_stream.mean(dim=-1, keepdims=True)
            q_values = value_stream + advantage_stream
        else:
            q_values = self.head(x.view(x.size(0), -1))
        return q_values