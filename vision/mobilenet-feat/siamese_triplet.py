"""
Training a face matching network using the triplet loss.
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from datasets import FaceTripletDataset
from util import get_1x_lr_params, get_10x_lr_params

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
MOMENTUM = 0.9
DATA_DIR = "./data/lfw"
WEIGHT_DIR = "./weights"
WEIGHT_DECAY = 0.0005

parser = argparse.ArgumentParser(description="MobileNet Arguments")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
parser.add_argument("--momentum", type=float, default=MOMENTUM)
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularization parameter.")
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory of the dataset.")
parser.add_argument("--weight-dir", type=str, default=WEIGHT_DIR, help="Directory of the network weights.")

random.seed(1364)
torch.manual_seed(1364)
torch.cuda.manual_seed_all(1364)
np.random.seed(1364)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FaceNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.project = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1280, 128),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = F.relu6(x)
        x = self.project(x)
        return x


def train():
    args = parser.parse_args()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainloader = DataLoader(FaceTripletDataset(args.data_dir, 'train', transform),
                             batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model.features[:12].parameters():
        param.requires_grad = False
    model = FaceNet(model)
    model.train()

    optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model), 'lr': args.lr},
                                 {'params': get_10x_lr_params(model), 'lr': 10 * args.lr}],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    device = torch.device("cuda:1")
    model.to(device)
    loss_value = 0
    number_of_constraints = 0
    number_of_errors = 0
    SAVE_MODEL_EVERY = 500  # len(trainloader) // 10  # Take a snapsho at every SAVE_MODEL_EVERY iteration
    SHOW_LOSS_EVERY = 10

    # state = torch.load(os.path.join(args.weight_dir, 'face_matching', f'sgd_snapshot_epoch_{0}_iter_{627}.pth'))
    # optimizer.load_state_dict(state['optimizer'])
    # model.load_state_dict(state['state_dict'])

    for epoch in range(0, args.num_epochs):
        for i_iter, batch in enumerate(trainloader):
            img0, img1, img2, names = batch
            # print(names)
            img0 = img0.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)

            optimizer.zero_grad()
            embed0 = model(img0)
            embed1 = model(img1)
            embed2 = model(img2)
            embed0 = embed0 / torch.norm(embed0, dim=1, keepdim=True)
            embed1 = embed1 / torch.norm(embed1, dim=1, keepdim=True)
            embed2 = embed2 / torch.norm(embed2, dim=1, keepdim=True)

            margin = 0.2

            dist01 = torch.norm(embed0 - embed1, dim=1)
            dist02 = torch.norm(embed0 - embed2, dim=1)
            loss = ((dist01 + margin) > dist02) * (dist01 + margin - dist02)
            number_of_constraints += (loss != 0).sum().item()
            number_of_errors += (loss > margin).sum().item()
            denom = (loss != 0).sum().item() if (loss != 0).sum().item() > 0 else 1
            loss = loss.sum() / denom

            loss.backward()
            optimizer.step()
            loss_value += loss.data.item()

            if (i_iter + 1) % SHOW_LOSS_EVERY == 0:
                print(f'epoch: {epoch}, iteration: {i_iter + 1}, average_loss: {loss_value/SHOW_LOSS_EVERY}, '
                      f'number_of_constraints in {SHOW_LOSS_EVERY} iterations: {number_of_constraints}, '
                      f'number of errors in {SHOW_LOSS_EVERY} iterations: {number_of_errors}')
                loss_value = 0
                number_of_constraints = 0
                number_of_errors = 0
            if (i_iter + 1) % SAVE_MODEL_EVERY == 0 or (i_iter + 1) == len(trainloader):
                print('taking snapshot from the trained model so far...')
                state = {
                    'epoch': epoch,
                    'iteration': i_iter,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(args.weight_dir, 'face_matching', f'sgd_snapshot_epoch_{epoch}_iter_{i_iter + 1}.pth'))
        # After each epoch, shrink the learning rate:
        if (epoch + 1) % 50 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
    return model


model = train()
