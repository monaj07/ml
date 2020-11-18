import numpy as np
import random
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.datasets import TREC


seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train_model(model, iterator, optimiser, device, weight):
    model.train()
    model.to(device)
    total_loss = 0
    for idx, batch in enumerate(iterator):
        src, lbl = batch.text, batch.label
        output = model(src.to(device))
        loss = F.cross_entropy(output, lbl.long().view(-1).to(device),
                               weight=weight.to(device))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    print(f"Average training loss: {round(total_loss / len(iterator), 2)}")
    print('-------------------------------------------')


def evaluate_model(model, iterator, num_classes, device, weight):
    model.eval()
    model.to(device)
    total_loss = 0
    conf_mat = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            src, lbl = batch.text, batch.label
            output = model(src.to(device))
            loss = F.cross_entropy(output, lbl.long().view(-1).to(device),
                                   weight=weight.to(device))
            total_loss += loss.item()
            predictions = output.argmax(dim=-1).cpu()
            conf_mat += confusion_matrix(lbl.data.view(-1),
                                         predictions.data.view(-1),
                                         labels=range(num_classes))

    print(f"Average evaluation loss: {np.round(total_loss / len(iterator), 2)}")
    recall = conf_mat / conf_mat.sum(axis=0, keepdims=True)
    precision = conf_mat / conf_mat.sum(axis=1, keepdims=True)
    print(f'Average Recall: {np.diag(recall).mean()}')
    print(f'Average Precision: {np.diag(precision).mean()}')
    # print(conf_mat)
    print('-------------------------------------------')


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes,
                 n_filters, kernel_sizes, pad_idx, dropout_p=0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size,
                                            padding_idx=pad_idx)
        self.cnn_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                   out_channels=n_filters,
                                                   kernel_size=(fs, embedding_size))
                                         for fs in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * n_filters, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        emb = self.embedding_layer(x).unsqueeze(1)
        emb = self.dropout(emb)
        # print(emb.shape)
        cnn_outputs = [F.relu6(layer(emb)).squeeze(-1)
                       for layer in self.cnn_layers]
        # print(cnn_outputs[1].shape)
        pooling_outputs = [F.max_pool1d(cnn_out, cnn_out.shape[-1]).squeeze(-1)
                           for cnn_out in cnn_outputs]
        # print(pooling_outputs[1].shape)
        fc_in = torch.cat(pooling_outputs, dim=-1)
        fc_in = self.dropout(fc_in)
        fc_out = F.relu6(self.fc(fc_in))
        # fc_out = self.dropout(fc_out)
        output = self.output_layer(fc_out)
        return output


def predict_class(model, sentence, text_field, device, min_len=4):
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
    return output


if __name__ == "__main__":
    text_field = Field(tokenize='spacy', lower=True, batch_first=True)
    label_field = LabelField()
    train_data, test_data = TREC.splits(text_field, label_field, fine_grained=False)
    train_data, valid_data = train_data.split()
    text_field.build_vocab(train_data, max_size=25000,
                           vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
    label_field.build_vocab(train_data)
    print([label_field.vocab.itos[i] for i in range(6)])

    batch_size = 64
    n_epochs = 15
    embedding_size = 100
    hidden_size = 10
    num_classes = len(label_field.vocab)
    n_filters = 100
    kernel_sizes = (2, 3, 4)
    dropout_p = 0.1
    vocab_size = len(text_field.vocab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    unk_idx = text_field.vocab.stoi[text_field.unk_token]

    model = CNN(vocab_size, embedding_size, hidden_size, num_classes,
                n_filters, kernel_sizes, pad_idx, dropout_p=dropout_p)

    model.embedding_layer.weight.data.copy_(text_field.vocab.vectors)
    model.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
    model.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)

    optimiser = optim.Adam(model.parameters())

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size
    )

    # Due to the major class imbalance (insufficient data from class 6):
    class_weight = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])

    # Random evaluation check:
    evaluate_model(model, valid_iterator, num_classes, device, class_weight)

    for epoch in range(n_epochs):
        print(f'\nEpoch number {epoch}:')
        train_model(model, train_iterator, optimiser, device, class_weight)
        evaluate_model(model, valid_iterator, num_classes, device, class_weight)

    print("\nTEST EVALUATION: ")
    evaluate_model(model, test_iterator, num_classes, device, class_weight)

    # ---------------------------------------------
    sentence = "Who was the father of Hitler?"
    class_probs = predict_class(model, sentence, text_field, device, min_len=4)
    pred_class = class_probs.argmax().item()
    print(f'Predicted class is: {pred_class} = {label_field.vocab.itos[pred_class]}')

    sentence = "How many minutes are in six hundred and eighteen hours?"
    class_probs = predict_class(model, sentence, text_field, device, min_len=4)
    pred_class = class_probs.argmax().item()
    print(f'Predicted class is: {pred_class} = {label_field.vocab.itos[pred_class]}')

    sentence = "What continent is Bulgaria in?"
    class_probs = predict_class(model, sentence, text_field, device, min_len=4)
    pred_class = class_probs.argmax().item()
    print(f'Predicted class is: {pred_class} = {label_field.vocab.itos[pred_class]}')

    sentence = "What does WYSIWYG stand for?"
    class_probs = predict_class(model, sentence, text_field, device, min_len=4)
    pred_class = class_probs.argmax().item()
    print(f'Predicted class is: {pred_class} = {label_field.vocab.itos[pred_class]}')

    sentence = " ".join(['What', 'is', 'a', 'religion', '?'])
    class_probs = predict_class(model, sentence, text_field, device, min_len=4)
    pred_class = class_probs.argmax().item()
    print(f'Predicted class is: {pred_class} = {label_field.vocab.itos[pred_class]}')
