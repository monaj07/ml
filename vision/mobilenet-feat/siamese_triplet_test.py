"""
Training a face matching network using the triplet loss.
"""
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from datasets import FaceTripletDataset

BATCH_SIZE = 64
DATA_DIR = "./data/lfw"
WEIGHT_DIR = "./weights"

parser = argparse.ArgumentParser(description="MobileNet Arguments")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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


def test():
    args = parser.parse_args()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testloader = DataLoader(FaceTripletDataset(args.data_dir, 'test', transform),
                             batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    model = FaceNet(model)

    state = torch.load(os.path.join(args.weight_dir, 'face_matching', f'sgd_snapshot_epoch_0_iter_627.pth'))
    model.load_state_dict(state['state_dict'])

    device = torch.device("cuda:1")
    model.to(device)
    labels = []

    with torch.no_grad():
        for i_iter, batch in tqdm(enumerate(testloader), total=len(testloader), desc='Evaluating FaceNet on the test set', ncols=100, leave=False):
            img0, img1, img2, names = batch
            # print(names)
            img0 = img0.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)

            embed0 = model(img0)
            embed1 = model(img1)
            embed2 = model(img2)
            embed0 = embed0 / torch.norm(embed0, dim=1, keepdim=True)
            embed1 = embed1 / torch.norm(embed1, dim=1, keepdim=True)
            embed2 = embed2 / torch.norm(embed2, dim=1, keepdim=True)

            dist01 = torch.norm(embed0 - embed1, dim=1)
            dist02 = torch.norm(embed0 - embed2, dim=1)
            label = ((dist01 - dist02) > 0).int().tolist()
            labels.extend(label)

    print(f"Number of errors: {sum(labels)} out of {len(labels)} cases.")


test()
