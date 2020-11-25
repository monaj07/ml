from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=2, dropout_p=0.5):
        super().__init__()
        if not isinstance(hidden_sizes, (int, list, tuple)):
            raise ValueError("""hidden_sizes must be either an integer, 
                                or a list/tuple of integers.""")
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.mid_fcs = nn.ModuleList([nn.Linear(hidden_sizes[i],
                                                hidden_sizes[i+1])
                                      for i in range(len(hidden_sizes)-1)])
        self.out_fc = nn.Linear(hidden_sizes[-1], num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu6(self.fc1(x))
        x = self.dropout(x)
        for layer in self.mid_fcs:
            x = F.relu6(layer(x))
            x = self.dropout(x)
        out = self.out_fc(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        if not isinstance(hidden_sizes, (int, list, tuple)):
            raise ValueError("""hidden_sizes must be either an integer, 
                                or a list/tuple of integers.""")
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.mid_fcs = nn.ModuleList([nn.Linear(hidden_sizes[i],
                                                hidden_sizes[i + 1])
                                      for i in range(len(hidden_sizes) - 1)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_sizes[i+1], 0.8)
                                        for i in range(len(hidden_sizes) - 1)])
        self.out_fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu6(self.fc1(x))
        for idx, layer in enumerate(self.mid_fcs):
            x = layer(x)
            x = self.bn_layers[idx](x)
            x = F.relu6(x)
        out = self.out_fc(x)
        out = torch.tanh(out)
        return out


def train_gan(G, D, input_noise_dim, train_loader, optimizer_G, optimizer_D, device):
    for idx, batch in (enumerate(train_loader)):
        batch_size = batch[0].shape[0]
        image_H = batch[0].shape[2]
        image_W = batch[0].shape[3]
        real_images = batch[0].to(device)

        G.to(device)
        D.to(device)

        # ------------------------------------
        # Training the Discriminator:
        # ------------------------------------
        # G.requires_grad_(False)
        # D.requires_grad_(True)

        # Generate fake images from uniform noise
        input_noise = torch.rand((batch_size, input_noise_dim)).to(device)
        fake_images = G(input_noise).view(batch_size, 1, image_H, image_W)

        # Combine real and fake images
        images = torch.cat([real_images, fake_images], dim=0)

        # Combine real + fake targets
        targets = torch.cat([torch.ones((batch_size, 1)),
                             torch.zeros((batch_size, 1))], dim=0).to(device)

        # Shuffle the real and fake data
        perm = torch.randperm(batch_size * 2)
        images = images[perm, ...].view(batch_size * 2, -1)
        targets = targets[perm, ...]

        # Compute loss function and update the discriminator parameters
        outputs = D(images)
        optimizer_D.zero_grad()
        loss = F.cross_entropy(outputs, targets.long().view(-1))
        loss.backward()
        optimizer_D.step()

        # ------------------------------------
        # Training the Generator:
        # ------------------------------------
        # G.requires_grad_(True)
        # D.requires_grad_(False)

        # Generate fake images from uniform noise
        input_noise = torch.rand((batch_size, input_noise_dim)).to(device)
        fake_images = G(input_noise).view(batch_size, 1, image_H, image_W)

        # Generate fake targets (set their expected target to one,
        # such that the parameters of G are updated
        # in order to generate a discriminator output close to one
        # (i.e. real class)
        targets = torch.ones((batch_size, 1)).to(device)

        # Compute loss function and update the generator parameters
        outputs = D(fake_images.view(batch_size, -1))
        optimizer_G.zero_grad()
        loss = F.cross_entropy(outputs, targets.long().view(-1))
        loss.backward()
        optimizer_G.step()

        # if idx % 1000 == 0:
        #     fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        #     for i in range(4):
        #         for j in range(4):
        #             ax[i, j].imshow(fake_images[i * 4 + j, 0, ...].cpu().data)
        #     plt.show()

    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(fake_images[i * 4 + j, 0, ...].cpu().data)
    plt.show()


if __name__ == "__main__":
    input_size_gen = 100
    hidden_sizes_gen = [128, 256, 512, 1024]
    hidden_sizes_dis = [512, 256]
    data_dim = 784  # 28 * 28
    batch_size_gen = 64
    n_epochs = 50
    lr = 0.0002
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Instantiate the Generator and Discriminator
    gen = Generator(input_size_gen, hidden_sizes_gen, data_dim)
    dis = Discriminator(data_dim, hidden_sizes_dis)

    # # Generate an example fake data and pass it through the discriminator
    # input_noise = torch.rand(input_size_gen)
    # fake_data = gen(input_noise)
    # dis_out = dis(fake_data)

    train_dataset = torchvision.datasets.MNIST('../data/',
                                               train=True, download=True,
                                               transform=transforms.Compose(
                                                   [transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        [0.5], [0.5])
                                                    ])
                                               )
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size_gen)

    # for idx, batch in enumerate(train_dataloader):
    #     images, labels = batch
    #     print(images.shape, labels.shape)
    # print(fake_data)
    # print(dis_out)

    optimizer_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        print(f"\n epoch {epoch}:")
        train_gan(gen, dis, input_size_gen, train_dataloader,
                  optimizer_G, optimizer_D, device)

