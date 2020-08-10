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


class TimeSeriesDatasetMature(data.Dataset):
    def __init__(self, input_data, input_width, label_width, shift, feature_columns, label_columns):
        window_size = input_width + shift
        num_samples = input_data.shape[0] - window_size
        self.regressors = []
        self.output_labels = []
        self.input_labels = []
        label_data = input_data[label_columns]
        regressors = input_data[feature_columns]
        for i in range(num_samples):
            regressor = regressors.iloc[i:i+window_size, :]
            future_labels = label_data.iloc[i+window_size-label_width:i+window_size]
            past_labels = label_data.iloc[i:i+input_width]
            self.regressors.append(regressor.to_numpy())
            self.output_labels.append(future_labels.to_numpy())
            self.input_labels.append(past_labels.to_numpy())

    def __len__(self):
        return len(self.output_labels)

    def __getitem__(self, idx):
        output_label = self.output_labels[idx]
        input_label = self.input_labels[idx]
        regressor = self.regressors[idx]
        return regressor, input_label, output_label