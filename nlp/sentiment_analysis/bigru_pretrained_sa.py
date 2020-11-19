"""
Changes w.r.t. the simple_sa.py version:
1. Added pre-trained Glove weights to the vocabulary (DONE)
    * Note: in order to avoid overfitting,
    in this case the learning rate should be much smaller than before,
    hence we just remove the manual setting of learning rate and
    leave it to Adam to use its own desired learning rate.
2 & 3. Bidirectional RNN + Multi-Layer [DONE]
    * Note: The final hidden state,
    has a shape of [num layers * num directions, batch size, hid dim].
    These are ordered: [forward_layer_0, backward_layer_0,
                        forward_layer_1, backward_layer 1, ...,
                        forward_layer_n, backward_layer n].
    As we want the final (top) layer forward and backward hidden states,
    we get the top two hidden layers from the first dimension,
    hidden[-2,:,:] and hidden[-1,:,:],
    and concatenate them together before passing them to the linear layer
    (after applying dropout).
    Reference: shorturl.at/mnAKS
4. Add DropOut to RNN [DONE]
"""

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
                 hidden_size, num_classes, pad_idx, bidirectional=True,
                 dropout_p=0.5, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size,
                                            padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(embedding_size, hidden_size, dropout=dropout_p,
                          bidirectional=bidirectional, num_layers=num_layers)
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, src):
        emb = self.dropout(self.embedding_layer(src))
        all_hs, h = self.rnn(emb)
        batch_size = src.shape[1]
        if self.bidirectional:
            h = h[-2:, :, :].permute(1, 0, 2).reshape(batch_size, -1)
        else:
            h = h[-1:, :, :].permute(1, 0, 2).reshape(batch_size, -1)
        # print(h.shape)
        # h = self.dropout(h)
        out = self.fc(h)
        return out


def predict_sentiment(model, sentence, text_field, device):
    model.eval()
    model.to(device)
    sentence = text_field.tokenize(sentence)
    tensor = torch.tensor([text_field.vocab.stoi[word] for word in sentence])
    print(tensor.shape)
    tensor = tensor.unsqueeze(1).to(device)
    output = model(tensor)
    output = F.softmax(output, dim=-1)
    print(output)


if __name__ == "__main__":
    text_field = Field(tokenize='spacy', lower=True)
    label_field = Field(dtype=torch.float, pad_token=None, unk_token=None)

    train_data, test_data = IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split()
    print(f'Number of training examples: {len(train_data.examples)}')
    print(f'Number of validation examples: {len(valid_data.examples)}')
    print(f'Number of test examples: {len(test_data.examples)}')

    text_field.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d',
                           unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)

    n_epochs = 5
    batch_size = 64
    embedding_size = 100
    hidden_size = 256
    num_layers = 1
    dropout_p = 0.5
    bidirectional = True
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    vocab_size = len(text_field.vocab)
    num_classes = len(label_field.vocab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = RNN(vocab_size, embedding_size, hidden_size, num_classes,
                bidirectional=bidirectional, pad_idx=pad_idx,
                dropout_p=dropout_p, num_layers=num_layers)

    # copying the pre-trained word embeddings into the embedding layer of our model.
    pretrained_embeddings = text_field.vocab.vectors
    print(pretrained_embeddings.shape)
    model.embedding_layer.weight.data.copy_(pretrained_embeddings)
    model.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
    model.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)

    model.to(device)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size,
        sort_key=lambda x: len(x.text), sort_within_batch=True
    )

    optimizer = optim.Adam(model.parameters())

    # Random evaluation check:
    evaluate_model(model, valid_iterator, device, num_classes)

    for epoch in range(n_epochs):
        print(f'\nEpoch number {epoch}:')
        train_model(model, train_iterator, device, optimizer)
        evaluate_model(model, valid_iterator, device, num_classes)

    print("\nTEST EVALUATION: ")
    evaluate_model(model, test_iterator, device, num_classes)

    print("Test on a sample sentence:")
    sentence = "The was movie super beautiful and awesome"
    predict_sentiment(model, sentence, text_field, device)
