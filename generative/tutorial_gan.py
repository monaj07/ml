import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Generator(nn.Module):
    def __init__(self, latent_dim=5, output_dim=2, hidden_sizes=(10, 5)):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(latent_dim, hidden_sizes[0]),
                       nn.BatchNorm1d(hidden_sizes[0]),
                       nn.ReLU6(inplace=True)])
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = torch.sigmoid(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_sizes=(10, 5), dropout_p=0.2):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(input_dim, hidden_sizes[0]),
                       nn.LeakyReLU(0.2, inplace=True),
                       nn.Dropout(dropout_p)])
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RealDataGenerator(Dataset):
    def __init__(self, numpy_data, maxim=None, minim=None):
        super().__init__()
        if minim is not None:
            numpy_data = numpy_data - minim
        if maxim is not None:
            numpy_data = numpy_data / (maxim - minim)
        self.data = numpy_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx, :]
        return item


def train_gan(generator, discriminator, latent_dim, dataloader,
              optimizer_G, optimizer_D, device, n_epochs,
              synthetic_data_normalised):
    for epoch in range(n_epochs + 1):
        # print(f"\n epoch {epoch}:")
        for idx, batch in enumerate(dataloader):
            bs = batch.shape[0]
            real_data = batch.to(device)
            real_data_labels = torch.ones(bs, 1).to(device)

            generator.to(device)
            discriminator.to(device)

            # ------------------------------------
            # Training the Discriminator:
            # ------------------------------------
            # generator.requires_grad_(False)
            # discriminator.requires_grad_(True)

            # generate fake data:
            input_noise = torch.rand(bs, latent_dim).to(device)
            fake_data = generator(input_noise)
            fake_data_true_labels = torch.zeros(bs, 1).to(device)

            # Combine reak and fake data into one training batch
            data_combined = torch.cat([real_data.double(),
                                       fake_data.double()], dim=0)
            labels_combined = torch.cat([real_data_labels, fake_data_true_labels], dim=0)

            # Shuffle the real and fake data
            perm = torch.randperm(bs * 2)
            labels_combined = labels_combined[perm, :]
            data_combined = data_combined[perm, :]

            # Check the discriminator output and update its parameters
            optimizer_D.zero_grad()
            discriminator_output = discriminator(data_combined.float())
            loss = F.cross_entropy(discriminator_output, labels_combined.long().view(-1))
            loss.backward()
            optimizer_D.step()

            # ------------------------------------
            # Training the Generator:
            # ------------------------------------
            # G.requires_grad_(True)
            # D.requires_grad_(False)

            input_noise = torch.rand(bs, latent_dim).to(device)
            fake_data = generator(input_noise)
            fake_data_fake_labels = torch.ones(bs, 1).to(device)

            optimizer_G.zero_grad()
            discriminator_output = discriminator(fake_data.float())
            loss = F.cross_entropy(discriminator_output, fake_data_fake_labels.long().view(-1))
            loss.backward()
            optimizer_G.step()
        if epoch % 50 == 0:
            input_noise = torch.rand(synthetic_data_normalised.shape[0],
                                     latent_dim).to(device)
            fake_data = generator(input_noise)
            batch_numpy = fake_data.cpu().data
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].plot(synthetic_data_normalised[:, 0],
                       synthetic_data_normalised[:, 1],
                       '.', markersize=3, marker='*', color='b')
            ax[0].set_title('Real Data')
            ax[1].plot(batch_numpy[:, 0], batch_numpy[:, 1],
                       '.', markersize=3, marker='s', color='k')
            ax[1].set_title(f'Generated data at epoch {epoch}')
            plt.show()


if __name__ == "__main__":
    sizes = [1000, 1000, 1000]
    mu0 = np.array([-13, 15])
    cov0 = np.array([[6, 5], [5, 6]])
    data0 = np.random.multivariate_normal(mu0, cov0, size=sizes[0])
    mu1 = np.array([0, -3])
    cov1 = np.array([[20, 0], [0, 1]])
    data1 = np.random.multivariate_normal(mu1, cov1, size=sizes[1])
    mu2 = np.array([10, 13])
    cov2 = np.array([[3, 0], [0, 3]])
    data2 = np.random.multivariate_normal(mu2, cov2, size=sizes[2])

    selected_data = [data0, data1, data2]

    synthetic_data = np.vstack(selected_data)
    labels = np.concatenate([i * np.ones(selected_data[i].shape[0])
                             for i in range(len(selected_data))]).astype(int)

    data_max = synthetic_data.max(axis=0, keepdims=1)
    data_min = synthetic_data.min(axis=0, keepdims=1)
    synthetic_data_normalised = (synthetic_data - data_min) / (data_max - data_min)
    batch_size = 128
    dataloader = DataLoader(RealDataGenerator(synthetic_data,
                                              data_max,
                                              data_min),
                            batch_size=batch_size, shuffle=True)

    lr = 0.0002
    n_epochs = 500
    latent_dim = 5
    output_dim = synthetic_data.shape[1]
    hidden_sizes_gen = (100, 50, 50)
    hidden_sizes_dis = (100, 50, 50)
    dropout_p = 0.1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Instantiate the Generator and Discriminator
    generator = Generator(latent_dim, output_dim, hidden_sizes_gen)
    discriminator = Discriminator(output_dim, hidden_sizes_dis, dropout_p)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    train_gan(generator, discriminator, latent_dim, dataloader,
              optimizer_G, optimizer_D, device, n_epochs,
              synthetic_data_normalised)