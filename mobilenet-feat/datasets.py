"""
RandomTestDataset dataloader to be used for generating random test input images in test_imagenet.py.
FacesDataset to be used for training/testing on Face dataset.
"""
import itertools
import glob
import os
import random
import numpy as np
import PIL
from torch.utils import data


class RandomTestDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = glob.glob(os.path.join(path, 'randomTest', 'images', '*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = PIL.Image.open(image_file)
        if self.transform:
            image = self.transform(image)
        return image, image_file


class FacesDataset(data.Dataset):
    def __init__(self, path, split='train', transform=None, used_data_ratio=1):
        self.path = path
        self.transform = transform
        self.image_files_face = glob.glob(os.path.join(path, 'CaltechFaces', split, 'Face', '*.jpg'))
        self.image_files_noface = glob.glob(os.path.join(path, 'CaltechFaces', split, 'NoFace', '**', '*.jpg'))
        len1 = int(len(self.image_files_face) * used_data_ratio)
        len2 = int(len(self.image_files_noface) * used_data_ratio)
        self.image_files = self.image_files_face[:len1] + self.image_files_noface[:len2]
        self.labels = np.hstack([np.ones(len(self.image_files_face[:len1])), np.zeros(len(self.image_files_noface[:len2]))]).astype(np.int)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_file = self.image_files[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, image_file


class FaceTripletDataset(data.Dataset):
    def __init__(self, path, split='train', transform=None, used_data_ratio=1):
        self.transform = transform
        persons = os.listdir(os.path.join(path, split))
        persons_files = []
        for p in persons:
            persons_files.append(os.listdir(os.path.join(path, split, p)))
        persons_len = len(persons)
        self.triplets = []
        for t in range(persons_len):
            neg_list = list(set(range(persons_len)) - {t})
            pairs = list(itertools.combinations(persons_files[t], 2))
            for pair in pairs:
                neg_t = random.choice(neg_list)
                neg_sample = random.choice(persons_files[neg_t])
                triplet = (os.path.join(path, split, persons[t]), *pair, os.path.join(path, split, persons[neg_t]), neg_sample)
                self.triplets.append(triplet)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        img0 = PIL.Image.open(os.path.join(triplet[0], triplet[1])).convert('RGB')
        img1 = PIL.Image.open(os.path.join(triplet[0], triplet[2])).convert('RGB')
        img2 = PIL.Image.open(os.path.join(triplet[3], triplet[4])).convert('RGB')
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img0, img1, img2, (triplet[1], triplet[2], triplet[4])
