"""
This code uses MobileNetv2, which is pre-trained on ImageNet dataset,
to classify random images into one of the 1000 categories defined in ImageNet.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from datasets import RandomTestDataset

HOME = os.getcwd()
BATCH_SIZE = 1
DATA_DIR = "data"
WEIGHT_DIR = None

parser = argparse.ArgumentParser(description="MobileNet Arguments")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory of the dataset.")
parser.add_argument("--weight-dir", type=str, default=WEIGHT_DIR, help="Directory of the network weights.")


def test():
    args = parser.parse_args()
    model = torchvision.models.mobilenet_v2(pretrained=True)
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    test_loader = DataLoader(RandomTestDataset(args.data_dir, transform),
                             batch_size=args.batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    model.eval()
    imagenet_labels = dict(pd.read_csv('imagenet_labels.csv').values)
    for i_iter, batch in enumerate(test_loader):
        image, image_file = batch
        image = image.to(device)
        output = model(image)
        out = output.cpu().data.numpy()
        predicted_label = np.argmax(out)
        print(f'{image_file}: {imagenet_labels[predicted_label]}')


test()
