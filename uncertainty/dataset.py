import glob
import os
import PIL
from torch.utils import data


class CreateDataBatches(data.Dataset):
    def __init__(self, input_data, labels, data_mean, data_std, normalize=False):
        self.input_data = input_data
        self.labels = labels
        self.data_mean = data_mean
        self.data_std = data_std
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_data = self.input_data[idx, :]
        if self.normalize:
            input_data = (input_data - self.data_mean) / self.data_std
        return input_data, label


class RandomTestDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = [filename for filename in os.listdir(path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = PIL.Image.open(os.path.join(self.path, image_file))
        if self.transform:
            image = self.transform(image)
        return image, image_file


class TimeSeriesDataset(data.Dataset):
    def __init__(self, input_data, labels, normalize=False):
        self.input_data = input_data
        self.labels = labels
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_data = self.input_data[idx, :]
        if self.normalize:
            input_data = (input_data - self.data_mean) / self.data_std
        return input_data, label