"""
The quantized fine-tuned model (the output of quantize.py) is evaluated on the test split of Faces dataset.
"""
import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from datasets import FacesDataset

BATCH_SIZE = 16
DATA_DIR = "./data"
WEIGHT_DIR = "./weights"

parser = argparse.ArgumentParser(description="MobileNet Arguments")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory of the dataset.")
parser.add_argument("--weight-dir", type=str, default=WEIGHT_DIR, help="Directory of the network weights.")


def test():
    args = parser.parse_args()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testloader = DataLoader(FacesDataset(args.data_dir, 'test', transform),
                            batch_size=args.batch_size, shuffle=True)

    model = torch.jit.load(os.path.join(args.weight_dir, 'faces', 'quantized_snapshot_epoch_4_iter_649.pth'))
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    results = []

    for i_iter, batch in tqdm(enumerate(testloader)):
        images, labels, image_files = batch
        images = images.to(device)
        output = model(images)
        _, predictions = torch.max(output, dim=1)
        labels = labels.numpy().reshape(-1, 1)
        predictions = predictions.numpy().reshape(-1, 1)
        results.append(np.hstack([labels, predictions]))

    results = np.vstack(results)
    cm = confusion_matrix(results[:, 0], results[:, 1])
    print(cm / cm.sum(axis=1))


test()
