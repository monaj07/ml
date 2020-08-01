from torch.utils import data


class CreateDataBatches(data.Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_data = self.input_data[idx, :]
        return input_data, label