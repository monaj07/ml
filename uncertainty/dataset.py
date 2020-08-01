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