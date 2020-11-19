import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB
from sklearn.metrics import confusion_matrix


seed = 1364
random.seed(seed)
torch.manual_seed(seed)


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, iterator, device, num_classes):
    print("\nEvaluating the model:")
    print('-------------------------------------------')
    model.eval()
    total_loss = 0
    total_conf_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            src, trg = batch.text, batch.label
            output = model(src.to(device))
            loss = F.cross_entropy(output, trg.long().squeeze(0).to(device))
            total_loss += loss.item()
            output_labels = torch.argmax(output, dim=1).cpu().data
            total_conf_mat += confusion_matrix(trg.view(-1), output_labels.view(-1))

    print(f"Average loss: {np.round(total_loss/len(iterator), 2)}")
    recall = total_conf_mat / total_conf_mat.sum(axis=0, keepdims=True)
    precision = total_conf_mat / total_conf_mat.sum(axis=1, keepdims=True)
    print(f'Average Recall: {np.diag(recall).mean()}')
    print(f'Average Precision: {np.diag(precision).mean()}')
    print('-------------------------------------------')


def train_model(model, iterator, device, optimizer):
    print("\nTraining the model:")
    print('-------------------------------------------')
    model.train()
    total_loss = 0
    agg_loss = 0
    counter = 0
    for idx, batch in enumerate(iterator):
        src, trg = batch.text, batch.label
        output = model(src.to(device))
        optimizer.zero_grad()
        loss = F.cross_entropy(output, trg.long().squeeze(0).to(device))
        loss.backward()
        optimizer.step()
        agg_loss += loss.item()
        total_loss += loss.item()
        counter += 1

        # if idx % 100 == 0 and idx > 0:
        #     print(f'Average training loss over the past {counter} iterations: '
        #           f'{np.round(agg_loss/counter, 2)}')
        #     agg_loss = 0
        #     counter = 0

    print(f"Average loss: {np.round(total_loss/len(iterator), 2)}")
    print('-------------------------------------------')


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, num_classes, dropout_p=0.1, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, src):
        emb = self.dropout(self.embedding_layer(src))
        all_hs, h = self.rnn(emb)
        assert torch.equal(h.squeeze(0), all_hs[-1, :, :])
        out = self.fc(h.squeeze(0))
        return out


if __name__ == "__main__":
    text_field = Field(tokenize='spacy', lower=True)
    label_field = Field(dtype=torch.float, pad_token=None, unk_token=None)

    train_data, test_data = IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split()
    print(f'Number of training examples: {len(train_data.examples)}')
    print(f'Number of validation examples: {len(valid_data.examples)}')
    print(f'Number of test examples: {len(test_data.examples)}')

    text_field.build_vocab(train_data, max_size=25000)
    label_field.build_vocab(train_data)

    n_epochs = 5
    batch_size = 64
    embedding_size = 100
    hidden_size = 256
    lr = 0.001
    vocab_size = len(text_field.vocab)
    num_classes = len(label_field.vocab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = RNN(vocab_size, embedding_size, hidden_size, num_classes)
    model.to(device)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size,
        sort_key=lambda x: len(x.text), sort_within_batch=True
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Random evaluation check:
    evaluate_model(model, valid_iterator, device, num_classes)

    for epoch in range(n_epochs):
        print(f'\nEpoch number {epoch}:')
        train_model(model, train_iterator, device, optimizer)
        evaluate_model(model, valid_iterator, device, num_classes)

    print("\nTEST EVALUATION: ")
    evaluate_model(model, test_iterator, device, num_classes)

