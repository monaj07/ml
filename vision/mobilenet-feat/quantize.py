"""
The fine-tuned quantizable model (the output of train_faces_quant.py) is quantized to int8 precision.
More info in:
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
"""
import os
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory of the dataset.")
parser.add_argument("--weight-dir", type=str, default=WEIGHT_DIR, help="Directory of the network weights.")
args = parser.parse_args()


def quantize(model_name=f'snapshot_final.pth'):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainloader = DataLoader(FacesDataset(args.data_dir, 'train', transform),
                            batch_size=args.batch_size, shuffle=True)

    state = torch.load(os.path.join(args.weight_dir, 'faces', model_name))
    model = torchvision.models.quantization.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 2, bias=True)
    model.load_state_dict(state['state_dict'])

    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    model.fuse_model()
    # model.qconfig = torch.quantization.default_qconfig
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    # Model Calibration
    for i_iter, batch in tqdm(enumerate(trainloader)):
        if i_iter > 100:
            # Enough with the calibration
            break
        images, labels, image_files = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = F.cross_entropy(output, labels)

    torch.quantization.convert(model, inplace=True)

    torch.jit.save(torch.jit.script(model),
                   os.path.join(args.weight_dir, 'faces', f'quantized_snapshot_epoch_{state["epoch"]}_iter_{state["iteration"] + 1}.pth'))

