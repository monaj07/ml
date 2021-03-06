"""
Fine-tuning the MobileNetv2 (pre-trained on ImageNet) for the task of Face/Person vs. non-person classification.
Only the last dense layer of the network is fine-tuned and the rest of the model is freezed to pre-trained weights.
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from datasets import FacesDataset

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DATA_DIR = "./data"
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


def train():
    args = parser.parse_args()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainloader = DataLoader(FacesDataset(args.data_dir, 'train', transform),
                             batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = torch.nn.Linear(1280, 2, bias=True)
    model.train()

    optimizer = torch.optim.SGD(model.classifier.parameters(),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    loss_value = 0
    SAVE_MODEL_EVERY = 100  # len(trainloader) // 10  # Take a snapsho at every SAVE_MODEL_EVERY iteration
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
                state = {
                    'epoch': epoch,
                    'iteration': i_iter,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                if not os.path.exists(os.path.join(args.weight_dir, 'faces')):
                    os.makedirs(os.path.join(args.weight_dir, 'faces'))
                torch.save(state, os.path.join(args.weight_dir, 'faces', f'snapshot_epoch_{epoch}_iter_{i_iter + 1}.pth'))
    return model


model = train()
