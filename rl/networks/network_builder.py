"""
This module makes creating deep neural networks (including CNNs) easy.
Implemented by Monaj at 3/10/2021.
"""
import torch.nn as nn


activation_functions = {'relu': nn.ReLU(), 'relu6': nn.ReLU6(), 'elu': nn.ELU(), 'l_relu': nn.LeakyReLU()}


class CreateNet(nn.Module):
    """
    This class is a helper for creating networks using pytorch one-liners.
    Example:
        ----------------------------------------------------------------
        In order to create a CNN with two conv layers,
        the first one mapping 3 channels to 16 channels and
        the 2nd one mapping 16 channels to 16 channels,
        and both with kernel_size=5 and stride=2,
        followed by two dense layers, one with 8 neurons and
        the 2nd one with two neurons (number of output classes),
        we write as following:
        (Assumptions in this example: input image dimension is (40, 90),
        and we have batchnorm and Relu, but not Dropout.)
        ------------------------------
        network_params = {
            'input_dim': (40, 90),
            'conv_layers': [(3, 16, 5, 2), (16, 16, 5, 2)],
            'dense_layers': [8, 2],
            'conv_bn': True,
            'activation': 'relu'
        }
        model = CreateNet(network_params)
        ----------------------------------------------------------------
        If for instance you want to add Dropout to the first dense layer:
        network_params = {
            'input_dim': (40, 90),
            'conv_layers': [(3, 16, 5, 2), (16, 16, 5, 2)],
            'dense_layers': [8, 2],
            'dense_dropout_p': 0.2,
            'conv_bn': True,
            'activation': 'relu'
        }
        ----------------------------------------------------------------
        Execute this file to run the unittest at the end of the file.
    """
    def __init__(self, params):
        super(CreateNet, self).__init__()

        # if there are conv layers in the network,
        # input_dim is going to be a tuple (h, w),
        # so the value of dense_input_shape will be re-written
        # inside the conv block
        try:
            input_dim = params['input_dim']
        except:
            raise ValueError("No input dimension is provided for the network")
        dense_input_shape = input_dim

        activation = params.get('activation', 'relu6')
        conv_bn = params.get('conv_bn', False)
        dense_bn = params.get('dense_bn', False)
        conv_dropout_p = params.get('conv_dropout_p', 0)
        dense_dropout_p = params.get('dense_dropout_p', 0)

        # computing the shape of the conv feature maps
        # (to be used when connecting their outputs to dense layers)
        def conv2d_shape_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        network_layers = []

        # Packing the conv layers in the network layers
        if 'conv_layers' in params.keys():
            conv_h, conv_w = input_dim
            for conv_layer in params['conv_layers']:
                in_channels, out_channels, kernel_size, stride = conv_layer
                network_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
                if conv_bn:
                    network_layers.append(nn.BatchNorm2d(out_channels))
                network_layers.append(activation_functions[activation])
                if conv_dropout_p > 0:
                    network_layers.append(nn.Dropout2d(conv_dropout_p))
                conv_h = conv2d_shape_out(conv_h, kernel_size=kernel_size, stride=stride)
                conv_w = conv2d_shape_out(conv_w, kernel_size=kernel_size, stride=stride)
            dense_input_shape = conv_w * conv_h * out_channels
            network_layers.append(nn.Flatten())

        # Packing dense layers in the network layers
        if 'dense_layers' in params.keys():
            for layer_idx, dense_layer_size in enumerate(params['dense_layers']):
                network_layers.append(nn.Linear(dense_input_shape, dense_layer_size))
                if layer_idx < (len(params['dense_layers']) - 1):
                    # We are NOT at the last layer of the network yet,
                    # so BN, Activation, and DropOut layers can be added.
                    if dense_bn:
                        network_layers.append(nn.BatchNorm1d(dense_layer_size))
                    network_layers.append(activation_functions[activation])
                    if dense_dropout_p > 0:
                        network_layers.append(nn.Dropout(dense_dropout_p))
                # The input to the next layer is the output of this layer
                dense_input_shape = dense_layer_size
        else:
            raise NotImplementedError('The network must have at least one dense layer')

        # Add layers parametrically to the network
        self.layers = nn.Sequential(*network_layers)

    def forward(self, x):
        output = self.layers(x)
        return output


if __name__ == "__main__":
    network_params = {
        'input_dim': (40, 90),
        'conv_layers': [(3, 16, 5, 2), (16, 16, 5, 2)],
        'dense_layers': [8, 2],
        'conv_bn': True,
        'activation': 'relu'
    }
    model = CreateNet(network_params)
    print(model)
