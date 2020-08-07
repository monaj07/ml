import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, nclasses, hidden_layers=None, dropout=0, dropout_input=0, batch_norm=False):
        super().__init__()
        self.init_complete = False
        self.input_size = input_size
        self.nclasses = nclasses
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.batch_norm = batch_norm
        self.reconstruct()
        self.init_complete = True

    def reconstruct(self):
        layers = self.__class__._collate_layers(self.input_size, self.nclasses, self.hidden_layers,
                                                self.dropout, self.dropout_input, self.batch_norm)
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _collate_layers(fs, nclasses, hidden_layers, dropout, dropout_input, batch_norm):
        layers = [nn.Dropout(dropout_input)] if dropout_input else []
        for idx in range(len(hidden_layers)):
            input_size = fs if idx == 0 else hidden_layers[idx-1]
            layers.append(nn.Linear(input_size, hidden_layers[idx]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[idx]))
            layers.append(nn.ReLU6())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layers[-1], nclasses))
        return layers

    def forward(self, x):
        x = self.net(x)
        return x

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, x):
        if x is None:
            self._hidden_layers = [8]
        else:
            try:
                x = [int(item) for item in x]
                check_pos_int = all(item>0 for item in x)
                if not check_pos_int:
                    raise ValueError("hidden_layers elements should be positive.")
                self._hidden_layers = x
            except:
                print("hidden_layer argument should be a 1D iterable of positive integers.")
                print("hidden_layers is set to [8].")
                self._hidden_layers = [8]
        if self.init_complete:
            self.reconstruct()

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, x):
        try:
            x = float(x)
            if x >= 1 or x < 0:
                raise ValueError(f"Dropout rate should be a float value x: (0 =< x < 1).")
            self._dropout = x
        except:
            print("Dropout rate should be a float value x: (0 =< x < 1).")
            print("Dropout is set to 0")
            self._dropout = 0
        if self.init_complete:
            self.reconstruct()


class MLR(MLP):
    def __init__(self, input_size, nclasses, hidden_layers=None, dropout=0, dropout_input=0, batch_norm=False):
        super().__init__(input_size, nclasses, hidden_layers, dropout, dropout_input, batch_norm)

    @staticmethod
    def _collate_layers(fs, nclasses, hidden_layers, dropout, dropout_input, batch_norm):
        layers = [nn.Dropout(dropout_input)] if dropout_input else []
        for idx in range(len(hidden_layers)):
            input_size = fs if idx == 0 else hidden_layers[idx-1]
            layers.append(nn.Linear(input_size, hidden_layers[idx]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[idx]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layers[-1], nclasses))
        return layers
