"""
In this code, I have implemented this paper:
Pruning Filters for Efficient Convnets, ICLR 2017.
By running some crude experiments, I got similar observations to the ones in Fig.2 of the paper,
which state that shallower layers are more sensitive to pruning compared to deeper filters.
Moreover, retraining the pruned network is necessary to heal the damaged network.
In fact, similar to what depicted in Fig.2, I managed to get no loss of accuracy,
even after pruning up to 90% of the weights in deeper layers.
By prunning further layers, we can get far more compression at a very negligible cost in accuracy. For example:
we can set PRUNE_RATIOS = [90, 85, 80, 75, 70, 65, 60, 55, 50, 20, 20, 20, 20, 20, 20],
where the left most element indicates the percentage of the channels dropped from the deepest layer.
The shallower layers are capped to a cut percentage of 20% and no less than that to avoid performance drop,
as noted above and denoted in the paper as well.

Assuming a model has been previously trained (e.g. using train_faces.py):
The prunning procedure is like this:
I) Prune the previously trained model, and see how much the model size is shrunk.
II) Test the pruned model, the performance is probably terrible, since a large portion of the weights have been removed.
III) Re-train the pruned model.
IV) Test the re-trained pruned model. The performance becomes comparable with that of the original model.
Then, go back to (I) and repeat the whole procedure multiple times until some criteria are met.
(e.g. until a sufficient model size, or reaching an accuracy limit)
"""

import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from datasets import FacesDataset

BATCH_SIZE = 16
DATA_DIR = "./data"
WEIGHT_DIR = "./weights"
NUM_EPOCHS = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

parser = argparse.ArgumentParser(description="MobileNet Arguments")

parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory of the dataset.")
parser.add_argument("--weight-dir", type=str, default=WEIGHT_DIR, help="Directory of the network weights.")
parser.add_argument("--lr", type=float, default=LEARNING_RATE)
parser.add_argument("--momentum", type=float, default=MOMENTUM)
parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularization parameter.")


def test():
    args = parser.parse_args()
    transform_test = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testloader = DataLoader(FacesDataset(args.data_dir, 'test', transform_test, used_data_ratio=1),
                            batch_size=args.batch_size, shuffle=False)

    state = torch.load(os.path.join(args.weight_dir, 'faces', 'snapshot_epoch_4_iter_649.pth'))
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 2, bias=True)
    model.load_state_dict(state['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    for pruning_plus_retraining_step in range(4):
        ##########################################################################################
        # PRUNING:
        PRUNING = True
        PRUNE_RATIOS = [90, 85, 80, 75, 70, 65, 60, 55, 50, 20, 20, 20, 20, 20, 20]

        if PRUNING:
            params = [np.array(params.size()).prod() for params in model.parameters()]
            model_size = sum(params)

            for ilayer, layer in enumerate(range(17, 2, -1)):
                orig_idx = model.features[layer].conv[0][0].weight.abs().sum(dim=1).view(-1).sort()[1]
                PRUNE_RATIO = PRUNE_RATIOS[ilayer]
                idx_cut = int(PRUNE_RATIO * len(orig_idx) / 100)
                idx = orig_idx[idx_cut:]

                conv = model.features[layer].conv[0][0]
                conv.out_channels = len(idx)
                conv.weight = torch.nn.Parameter(conv.weight[idx, :, :, :])

                bn1 = model.features[layer].conv[0][1]
                bn1.num_features = len(idx)
                bn1.running_mean = bn1.running_mean[idx]
                bn1.running_var = bn1.running_var[idx]
                bn1.weight = torch.nn.Parameter(bn1.weight[idx])
                bn1.bias = torch.nn.Parameter(bn1.bias[idx])

                convdw = model.features[layer].conv[1][0]
                convdw.in_channels = len(idx)
                convdw.out_channels = len(idx)
                convdw.weight = torch.nn.Parameter(convdw.weight[idx, :, :, :])
                convdw.groups = len(idx)

                bn2 = model.features[layer].conv[1][1]
                bn2.num_features = len(idx)
                bn2.running_mean = bn2.running_mean[idx]
                bn2.running_var = bn2.running_var[idx]
                bn2.weight = torch.nn.Parameter(bn2.weight[idx])
                bn2.bias = torch.nn.Parameter(bn2.bias[idx])

                conv2 = model.features[layer].conv[2]
                conv2.in_channels = len(idx)
                conv2.weight = torch.nn.Parameter(conv2.weight[:, idx, :, :])

            params = [np.array(params.size()).prod() for params in model.parameters()]
            pruned_model_size = sum(params)

            print(f"Original model size: {model_size}\n  Pruned model size: {pruned_model_size}\n")
        ##########################################################################################

        ##########################################################################################
        # Test the pruned model:
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
        print()
        print(cm)
        print(cm / cm.sum(axis=1)[:, np.newaxis])
        ##########################################################################################

        ##########################################################################################
        # Fine-tuning the pruned model:
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainloader = DataLoader(FacesDataset(args.data_dir, 'train', transform),
                                 batch_size=args.batch_size, shuffle=True)

        for param in model.parameters():
            param.requires_grad = True
        model.train()

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        model.to(device)
        loss_value = 0
        SAVE_MODEL_EVERY = 100
        SHOW_LOSS_EVERY = 10
        for epoch in range(args.num_epochs):
            for i_iter, batch in enumerate(trainloader):
                images, labels, image_files = batch
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels, reduction='mean')
                loss.backward()
                loss_value += loss.data.item()
                optimizer.step()

                if (i_iter + 1) % SHOW_LOSS_EVERY == 0:
                    print(f'epoch: {epoch}, iteration: {i_iter + 1}, average_loss: {loss_value/SHOW_LOSS_EVERY}')
                    loss_value = 0
                if (i_iter + 1) % SAVE_MODEL_EVERY == 0 or (i_iter + 1) == len(trainloader):
                    print('taking snapshot from the trained model so far...')
                    torch.save(model, os.path.join(args.weight_dir, 'faces', f'pruned_model_pruningStep_{pruning_plus_retraining_step}_epoch_{epoch}_iter_{i_iter + 1}.pth'))
        ##########################################################################################

        ##########################################################################################
        # Test the pruned+trained model:
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
        print()
        print(cm)
        print(cm / cm.sum(axis=1)[:, np.newaxis])
        ##########################################################################################

test()

