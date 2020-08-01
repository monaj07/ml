from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#----------------------
from dataset import CreateDataBatches
from models import Model
from utils_cm import compute_cm, split_dataset

### Generate synthetic data using Gaussians
sizes = [1000, 1000]

# class0:
mu0 = np.array([-3, 5])
cov0 = np.array([[10, 0], [0, 10]]) * 0.5
data0 = np.random.multivariate_normal(mu0, cov0, size=sizes[0])

# class1:
mu1 = np.array([0, -3])
cov1 = np.array([[20, -1], [-1, 20]]) * 0.5
data1 = np.random.multivariate_normal(mu1, cov1, size=sizes[1])

### Combine data from different classes, shuffle them and split it into train, validation and test sets
data = np.vstack([data0, data1])
labels = np.concatenate([i * np.ones(sizes[i]) for i in range(len(sizes))]).astype(int)
N = sum(sizes)
Ts = [int(v * N) for v in (0.6, 0.8, 1)]
data_train, labels_train, data_val, labels_val, data_test, labels_test = split_dataset(data, labels, Ts)

classes = np.unique(labels)
class_colours = ['r', 'b']

idx_train = [np.where(labels_train == c)[0] for c in classes]
idx_test = [np.where(labels_test == c)[0] for c in classes]

for c in classes:
    plt.scatter(data_train[idx_train[c], 0], data_train[idx_train[c], 1], marker='.', s=100, color=class_colours[c])

trainloader = DataLoader(CreateDataBatches(data_train, labels_train), batch_size=16, shuffle=True)
valloader = DataLoader(CreateDataBatches(data_val, labels_val), batch_size=16, shuffle=True)
testloader = DataLoader(CreateDataBatches(data_test, labels_test), batch_size=16, shuffle=True)

model = Model(input_size=data_train.shape[-1], nclasses=len(classes), hidden_layers=[8], dropout=0)

optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
model.to(device)
print(device)

loss_value = 0
n_epochs = 1
reset_loss_every = 10

model.train()
for epoch in range(n_epochs):
    for it, train_batch in enumerate(trainloader):
        model.train()
        train_data_batch, train_labels_batch = train_batch
        output = model(train_data_batch.to(device).float())
        optim.zero_grad()
        loss = F.cross_entropy(output, train_labels_batch.to(device), reduction="mean")
        loss.backward()
        loss_value += loss.data.item()
        optim.step()

        if it % reset_loss_every == 0 and it > 0:
            model.eval()
            gt_val, preds_val = [], []
            for it_val, val_batch in enumerate(valloader):
                val_data_batch, val_labels_batch = val_batch
                output_val = model(val_data_batch.to(device).float())
                preds_val.append(F.softmax(output_val, dim=1).data.numpy().argmax(axis=1))
                gt_val.append(val_labels_batch.numpy())
            preds_val = np.hstack(preds_val)
            gt_val = np.hstack(gt_val)
            recall, precision = compute_cm(gt_val, preds_val, classes)
            average_loss = np.round(loss_value / reset_loss_every, 4)
            print(f'epoch: {epoch}, iteration: {it}, recall: {recall},  precision: {precision}, average_loss: {average_loss}')
            loss_value = 0
