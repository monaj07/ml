import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, input_size, num_rnn_layers, hidden_size, seq_len, num_classes, device):
        super(RNN, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.GRU(input_size, hidden_size, num_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_rnn_layers, x.shape[0], self.hidden_size).to(self.device)

        out, _ = self.rnn(x, h0)
        # out = out.reshape(out.shape[0], -1)
        out = self.fc1(out[:, -1, :])
        return out


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
input_size = 28
seq_len = 28
num_rnn_layers = 2
hidden_size = 50
num_classes = 10
lr = 0.001
batch_size = 64
n_epochs = 2

data_train = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_test = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

model = RNN(input_size, num_rnn_layers, hidden_size, seq_len, num_classes, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
model.train()
for epoch in range(n_epochs):
    for idx, (data, targets) in tqdm(enumerate(data_train_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        # RNN expects a data of shape (B x seq_len x feature_size)
        # Here our data is a batch of images of the shape (B x 1 x 28 x 28)
        # so we just remove the 2nd dimension, and assume each row of the images as one time point
        # of the sequence, where each time point has 28 feature
        data = data.squeeze(dim=1)
        output = model(data)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()


# Evaluation
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('check accuracy on train set: ')
    else:
        print('check accuracy on test set: ')
    model.eval()
    check = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            # RNN expects a data of shape (B x seq_len x feature_size)
            # Here our data is a batch of images of the shape (B x 1 x 28 x 28)
            # so we just remove the 2nd dimension, and assume each row of the images as one time point
            # of the sequence, where each time point has 28 feature
            x = x.squeeze(dim=1)
            y_hat = model(x)
            check.append((y_hat.argmax(dim=1) == y).numpy())
    check = np.hstack(check)
    print(f'accuracy = {(100*sum(check)/len(check)).round(2)}')

check_accuracy(data_train_loader, model)
check_accuracy(data_test_loader, model)



