import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB


seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train_model(model, iterator, optimiser, device):
    model.train()
    model.to(device)
    total_loss = 0
    for idx, batch in enumerate(iterator):
        src, lbl = batch.text, batch.label
        output = model(src.to(device))
        loss = F.cross_entropy(output, lbl.long().view(-1).to(device))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    print(f"Average training loss: {round(total_loss / len(iterator), 2)}")
    print('-------------------------------------------')


def evaluate_model(model, iterator, num_classes, device):
    model.eval()
    model.to(device)
    total_loss = 0
    conf_mat = np.zeros((num_classes, num_classes))
    for idx, batch in enumerate(iterator):
        src, lbl = batch.text, batch.label
        output = model(src.to(device))
        loss = F.cross_entropy(output, lbl.long().view(-1).to(device))
        total_loss += loss.item()
        predictions = output.argmax(dim=-1).cpu()
        conf_mat += confusion_matrix(lbl.data.view(-1), predictions.data.view(-1))

    print(f"Average evaluation loss: {np.round(total_loss/len(iterator), 2)}")
    recall = conf_mat / conf_mat.sum(axis=0, keepdims=True)
    precision = conf_mat / conf_mat.sum(axis=1, keepdims=True)
    print(f'Average Recall: {np.diag(recall).mean()}')
    print(f'Average Precision: {np.diag(precision).mean()}')
    print('-------------------------------------------')


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_idx,
                 n_filters=100, filter_sizes=(2, 3, 5),
                 num_classes=2, dropout_p=0.1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size,
                                            padding_idx=pad_idx)
        self.cnn_filters = nn.ModuleList([nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(fs, embedding_size)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        emb = self.embedding_layer(x)
        emb = self.dropout(emb)
        cnn_outputs = [layer(emb.unsqueeze(1)).squeeze(-1)
                       for layer in self.cnn_filters]
        # each cnn_out shape:
        # [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooling_outputs = [F.max_pool1d(cnn_out, cnn_out.shape[-1])
                           for cnn_out in cnn_outputs]
        # each pooling_out shape: [batch size, n_filters, 1]
        relu_outputs = [F.relu6(pooling_out).squeeze(-1) for pooling_out in pooling_outputs]
        fc_in = torch.cat(relu_outputs, dim=-1)
        fc_out = self.fc(fc_in)
        fc_out = self.dropout(fc_out)
        outputs = self.output_layer(fc_out)
        return outputs


def predict_sentiment(model, sentence, text_field, device, min_len=5):
    model.eval()
    model.to(device)
    sentence = text_field.tokenize(sentence)
    if len(sentence) < min_len:
        sentence += text_field.pad_token * (min_len - len(sentence))
    tensor = torch.tensor([text_field.vocab.stoi[word] for word in sentence])
    tensor = tensor.unsqueeze(0).to(device)
    print(tensor.shape)
    output = model(tensor)
    output = F.softmax(output, dim=-1)
    print(output)


if __name__ == '__main__':
    text_field = Field(tokenize='spacy', lower=True, batch_first=True)
    label_field = Field(dtype=torch.float, batch_first=True,
                        pad_token=None, unk_token=None)

    train_data, test_data = IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split()

    text_field.build_vocab(train_data, max_size=25000,
                           vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)

    batch_size = 64
    n_epochs = 5
    embedding_size = 100
    hidden_size = 10
    n_filters = 100
    filters_size = (2, 3, 5)
    dropout_p = 0.5
    vocab_size = len(text_field.vocab)
    num_classes = len(label_field.vocab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]

    model = CNN(vocab_size, embedding_size, hidden_size, pad_idx, n_filters,
                filters_size, num_classes=num_classes, dropout_p=dropout_p)

    pretrained_weights = text_field.vocab.vectors
    model.embedding_layer.weight.data.copy_(pretrained_weights)
    model.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
    model.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size,
        sort_within_batch=True, sort_key=lambda x: len(x.text)
    )

    optimizer = optim.Adam(model.parameters())

    # Random evaluation check:
    evaluate_model(model, valid_iterator, num_classes, device)

    for epoch in range(n_epochs):
        print(f'\nEpoch number {epoch}:')
        train_model(model, train_iterator, optimizer, device)
        evaluate_model(model, valid_iterator, num_classes, device)

    print("\nTEST EVALUATION: ")
    evaluate_model(model, test_iterator, num_classes, device)

    print("Test on a sample sentence:")
    sentence = "The movie was super bad and aweful"
    predict_sentiment(model, sentence, text_field, device)

