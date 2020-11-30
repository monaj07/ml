"""
Alternative implementation:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(100, 50), latent_dim=20, dropout_p=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        layers = list()
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(dropout_p))
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_dims[-1], 2*latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.latent_dim, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim=20, hidden_dims=(50, 100), dropout_p=0.1):
        super().__init__()
        layers = list()
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # layers.append(nn.Dropout(dropout_p))
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        out = torch.sigmoid(z)
        return out


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


def train_gan(encoder, decoder, latent_dim, dataloader,
              optimizer_e, optimizer_d, device, n_epochs,
              synthetic_data_normalised):
    encoder.to(device)
    decoder.to(device)
    for epoch in range(n_epochs + 1):
        # print(f"\n epoch {epoch}:")
        encoder.train()
        decoder.train()
        for idx, batch in enumerate(dataloader):
            input_data = batch.to(device)

            z_mean_std = encoder(input_data.float())
            # The output of encoder is [BatchSize, latent_dim, 2]
            z_mean = z_mean_std[..., 0]
            z_std = torch.exp(0.5 * z_mean_std[..., 1])
            standard_normal_sample = torch.randn(latent_dim)
            z_std = z_std * standard_normal_sample
            z = z_mean + z_std

            optimizer_e.zero_grad()
            optimizer_d.zero_grad()

            output = decoder(z.squeeze()).double()
            # loss_recons = F.mse_loss(output, input_data.double(), reduction='sum')
            loss_recons = F.binary_cross_entropy(output, input_data.double(), reduction='sum')

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            loss_kld = -0.5 * torch.sum(1 + z_mean_std[..., 1] -
                                        z_mean.pow(2) -
                                        z_mean_std[..., 1].exp())

            loss = loss_recons + loss_kld
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()

        if epoch % 50 == 0:
            # decoder.eval()
            input_noise = torch.randn(synthetic_data_normalised.shape[0],
                                     latent_dim).to(device)
            fake_data = decoder(input_noise)
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

    lr = 0.002
    n_epochs = 500
    latent_dim = 5
    output_dim = synthetic_data.shape[1]
    hidden_sizes_encoder = (50, 100, 50, 20)
    hidden_sizes_decoder = (50, 100, 50, 20)
    dropout_p = 0.1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Instantiate the Generator and Discriminator
    encoder = Encoder(output_dim, hidden_sizes_encoder, latent_dim, dropout_p)
    decoder = Decoder(output_dim, latent_dim, hidden_sizes_decoder, dropout_p)

    optimizer_e = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))


    train_gan(encoder, decoder, latent_dim, dataloader,
              optimizer_e, optimizer_d, device, n_epochs,
              synthetic_data_normalised)
