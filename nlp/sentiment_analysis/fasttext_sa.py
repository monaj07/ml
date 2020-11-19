import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB


def add_bigrams(text):
    bigrams = set([" ".join(text[i: i+2]) for i in range(len(text)-1)])
    text_augm = text + list(bigrams)
    return text_augm


def train_model(model, iterator, device, optimiser):
    model.train()
    model.to(device)
    total_loss = 0
    for idx, batch in enumerate(iterator):
        src, lbl = batch.text, batch.label
        output = model(src.to(device))
        optimiser.zero_grad()
        loss = F.cross_entropy(output, lbl.long().squeeze(0).to(device))
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    print(f"Average training loss: {round(total_loss / len(iterator), 2)}")
    print('-------------------------------------------')


def evaluate_model(model, iterator, device, num_classes):
    model.eval()
    model.to(device)
    total_loss = 0
    conf_mat = np.zeros((num_classes, num_classes))
    for idx, batch in enumerate(iterator):
        src, lbl = batch.text, batch.label
        output = model(src.to(device))
        # print(output.shape, lbl.shape)
        loss = F.cross_entropy(output, lbl.long().squeeze(0).to(device))
        total_loss += loss.item()
        predictions = output.argmax(-1).cpu().data
        conf_mat += confusion_matrix(lbl.view(-1), predictions.view(-1))

    print(f"Average evaluation loss: {np.round(total_loss/len(iterator), 2)}")
    recall = conf_mat / conf_mat.sum(axis=0, keepdims=True)
    precision = conf_mat / conf_mat.sum(axis=1, keepdims=True)
    print(f'Average Recall: {np.diag(recall).mean()}')
    print(f'Average Precision: {np.diag(precision).mean()}')
    print('-------------------------------------------')


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 pad_idx, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size,
                                            padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding_layer(x)
        # emb = F.avg_pool2d(emb.permute(1, 0, 2), (emb.shape[0], 1)).squeeze(1)
        emb = emb.mean(dim=0)
        emb = self.dropout(emb)
        h = self.hidden_layer(emb)
        h = self.dropout(h)
        out = self.output_layer(h)
        return out


def predict_sentiment(model, sentence, text_field, device):
    model.eval()
    model.to(device)
    sentence = text_field.tokenize(sentence)
    sentence = add_bigrams(sentence)
    tensor = torch.tensor([text_field.vocab.stoi[word] for word in sentence])
    print(tensor.shape)
    tensor = tensor.unsqueeze(1).to(device)
    output = model(tensor)
    output = F.softmax(output, dim=-1)
    print(output)


seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":

    # Not sure really if adding bigrams in this problem helps that much;
    # You can test by removing preprocessing=add_bigrams argument
    # from the text_field and comparing the results
    text_field = Field(tokenize='spacy', lower=True, preprocessing=add_bigrams)
    label_field = Field(dtype=torch.float, pad_token=None, unk_token=None)

    train_data, test_data = IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split()
    text_field.build_vocab(train_data, max_size=25000,
                           vectors='glove.6B.100d',
                           unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)

    n_epochs = 5
    embedding_size = 100
    hidden_size = 10
    batch_size = 64
    dropout_p = 0.1
    num_classes = len(label_field.vocab)
    vocab_size = len(text_field.vocab)
    unk_idx = text_field.vocab.stoi[text_field.unk_token]
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size, sort_within_batch=True
    )

    model = FastText(vocab_size, embedding_size, hidden_size,
                     pad_idx=pad_idx, num_classes=num_classes, dropout_p=dropout_p)
    pretrained_weights = text_field.vocab.vectors
    model.embedding_layer.weight.data.copy_(pretrained_weights)
    model.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
    model.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)

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