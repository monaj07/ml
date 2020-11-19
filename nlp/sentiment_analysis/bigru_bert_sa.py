"""
Reference: shorturl.at/IJMVW
"""

import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import IMDB
from transformers import DistilBertTokenizer, DistilBertModel


seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
# You could also use 'bert-base-uncased' which is almost twice larger
# For that, BertTokenizer and BertModel should be imported

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes['distilbert-base-uncased']


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tokenize_and_trunc(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens


def evaluate_model(model, iterator, device, num_classes):
    print("\nEvaluating the model:")
    print('-------------------------------------------')
    model.eval()
    model.to(device)
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

    print(f"Average loss: {np.round(total_loss / len(iterator), 2)}")
    recall = total_conf_mat / total_conf_mat.sum(axis=0, keepdims=True)
    precision = total_conf_mat / total_conf_mat.sum(axis=1, keepdims=True)
    print(f'Average Recall: {np.diag(recall).mean()}')
    print(f'Average Precision: {np.diag(precision).mean()}')
    print('-------------------------------------------')


def train_model(model, iterator, device, optimizer):
    print("\nTraining the model:")
    print('-------------------------------------------')
    model.train()
    model.to(device)
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

    print(f"Average loss: {np.round(total_loss / len(iterator), 2)}")
    print('-------------------------------------------')


class BertGRU(nn.Module):
    def __init__(self, embedding_size, rnn_hidden_size,
                 num_classes, bidirectional=True, dropout_p=0.5):
        super().__init__()
        self.embedding_layer = bert
        self.rnn = nn.GRU(embedding_size, rnn_hidden_size,
                          bidirectional=bidirectional, batch_first=True)
        self.output_layer = nn.Linear(rnn_hidden_size * 2 if bidirectional
                                      else rnn_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.bidirectional = bidirectional

    def forward(self, x):
        # The output of DistilBert embedding is a tuple of size 1
        with torch.no_grad():
            emb = self.embedding_layer(x)[0]
        emb = self.dropout(emb)
        rnn_out, h = self.rnn(emb)
        batch_size = x.shape[0]

        if self.bidirectional:
            h = h[-2:, :, :].permute(1, 0, 2).reshape(batch_size, -1)
        else:
            h = h[-1:, :, :].permute(1, 0, 2).reshape(batch_size, -1)

        # if self.rnn.bidirectional:
        #     h = self.dropout(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))
        # else:
        #     h = self.dropout(h[-1, :, :])

        h = self.dropout(h)
        out = self.output_layer(h)
        return out


def predict_sentiment(model, sentence, tokenizer, device):
    model.eval()
    model.to(device)
    tokens = tokenizer.tokenize(sentence)
    indices = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens)
    indices += [eos_token_idx]
    tensor = torch.tensor(indices)
    print(tensor.shape)
    tensor = tensor.unsqueeze(0).to(device)
    output = model(tensor)
    output = F.softmax(output, dim=-1)
    print(output)


if __name__ == "__main__":
    text_field = Field(use_vocab=False,
                       tokenize=tokenize_and_trunc,
                       preprocessing=tokenizer.convert_tokens_to_ids,
                       batch_first=True,
                       init_token=init_token_idx,
                       eos_token=eos_token_idx,
                       pad_token=pad_token_idx,
                       unk_token=unk_token_idx)
    label_field = LabelField()

    train_data, test_data = IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split()
    label_field.build_vocab(train_data)

    n_epochs = 5
    batch_size = 128
    rnn_hidden_size = 256
    dropout_p = 0.2
    num_classes = len(label_field.vocab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = BertGRU(bert.config.to_dict()['dim'],
                    rnn_hidden_size, num_classes=num_classes,
                    dropout_p=dropout_p)

    for name, params in model.named_parameters():
        if name.startswith('embedding_layer'):
            params.requires_grad = False

    print(f"Number of trainable parameters: {get_model_size(model)}")

    optimiser = optim.Adam(model.parameters())

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size
    )

    # Random evaluation check:
    evaluate_model(model, valid_iterator, device, num_classes)

    for epoch in range(n_epochs):
        print(f'\nEpoch number {epoch}:')
        train_model(model, train_iterator, device, optimiser)
        evaluate_model(model, valid_iterator, device, num_classes)

    print("\nTEST EVALUATION: ")
    evaluate_model(model, test_iterator, device, num_classes)

    print("Test on a sample sentence:")
    sentence = "The was movie super bad and aweful"
    predict_sentiment(model, sentence, tokenizer, device)