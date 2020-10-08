import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
input_size = 784
hidden_size = 50
num_classes = 10
lr = 0.001
batch_size = 64
n_epochs = 2

data_train = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_test = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
model.train()
for epoch in range(n_epochs):
    for idx, (data, targets) in tqdm(enumerate(data_train_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        data = data.view(data.shape[0], -1)
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
            x = x.view(x.shape[0], -1)
            y_hat = model(x)
            check.append((y_hat.argmax(dim=1) == y).numpy())
    check = np.hstack(check)
    print(f'accuracy = {(100*sum(check)/len(check)).round(2)}')

check_accuracy(data_train_loader, model)
check_accuracy(data_test_loader, model)



