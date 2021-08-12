"""
Evaluating the MobileNetv2 (pre-trained on ImageNet and fine-tuned on the training split of the Face dataset)
on the test set of the Face dataset.
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

    state = torch.load(os.path.join(args.weight_dir, 'faces', 'snapshot_epoch_0_iter_649.pth'))
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 2, bias=True)
    model.load_state_dict(state['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    results = []

    for i_iter, batch in tqdm(enumerate(testloader)):
        images, labels, image_files = batch
        images = images.to(device)
        output = model(images)
        _, predictions = torch.max(output, dim=1)
        labels = labels.numpy().reshape(-1, 1)
        predictions = predictions.cpu().numpy().reshape(-1, 1)
        results.append(np.hstack([labels, predictions]))

    results = np.vstack(results)
    cm = confusion_matrix(results[:, 0], results[:, 1])
    print(cm / cm.sum(axis=1))


test()
