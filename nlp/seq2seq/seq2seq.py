import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from tqdm import tqdm


spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tokenize_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [token.text for token in spacy_eng.tokenizer(text)]


english_field = Field(sequential=True, use_vocab=True,
                      lower=True, tokenize=tokenize_eng,
                      init_token='<sos>', eos_token='<eos>')
german_field = Field(sequential=True, use_vocab=True,
                     lower=True, tokenize=tokenize_ger,
                     init_token='<sos>', eos_token='<eos>')
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"),
                                                    fields=(german_field, english_field))
german_field.build_vocab(train_data, max_size=10000, min_freq=2)
english_field.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_size, hidden_size, p):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, (h, c) = self.rnn(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, embedding_size, hidden_size, p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c):
        x = self.dropout(self.embedding(x))
        x, (h, c) = self.rnn(x, (h, c))
        x = self.fc(x)
        return x, h, c


class Seq2Seq(nn.Module):
    def __init__(self, german_field, english_field, num_layers, embedding_size, hidden_size, p):
        super(Seq2Seq, self).__init__()
        self.german_field = german_field
        self.english_field = english_field
        self.encoder = Encoder(len(german_field.vocab), num_layers, embedding_size, hidden_size, p)
        self.decoder = Decoder(len(english_field.vocab), num_layers, embedding_size, hidden_size, p)

    def forward(self, input_seq, target_seq, teacher_forcing=0.5):
        h, c = self.encoder(input_seq)
        output = []
        x = target_seq[0].unsqueeze(0)
        for it in range(1, len(target_seq)):
            out, h, c = self.decoder(x, h, c)
            if np.random.rand(1) < teacher_forcing:
                x = target_seq[it].unsqueeze(0)
            else:
                x = out.argmax(dim=-1)
            output.append(out)
        # Append a zero pre
        # Initial sos:
        init_sos = 0 * output[0]
        init_sos[:, :, 2] = 1
        # Append the init_sos to the beginning of the output vector
        output = [init_sos] + output
        output = torch.cat(output, dim=0)
        return output

    def translate(self, german_sentence, max_length=50):
        spacy_ger = spacy.load('de')
        tokenised_german_sentence = [tok.text.lower() for tok in spacy_ger.tokenizer(german_sentence)]
        tokenised_german_sentence.insert(0, german_field.init_token)
        tokenised_german_sentence.append(german_field.eos_token)
        tokenised_german_indices = [german_field.vocab.stoi[token] for token in tokenised_german_sentence]

        # Convert to Tensor
        tokenised_german_tensor = torch.LongTensor(tokenised_german_indices).unsqueeze(1).to(device)
        with torch.no_grad():
            sos = english_field.vocab.stoi['<sos>']
            dummy_target = torch.tensor([sos]*max_length).unsqueeze(1).to(device)
            output = self.forward(tokenised_german_tensor, dummy_target, teacher_forcing=0)
        output = output.argmax(dim=-1).squeeze()
        output_str = []
        for word in output:
            word = english_field.vocab.itos[word]
            if word == '<eos>':
                break
            output_str.append(word)
        output_str = " ".join(output_str[1:])
        return output_str



num_layers = 2
embedding_size = 300
hidden_size = 1024
p = 0.5
lr = 0.001
batch_size = 64
num_epochs = 100
EVERY = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(german_field, english_field, num_layers, embedding_size, hidden_size, p)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
pad_idx = english_field.vocab.stoi['<pad>']

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

german_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
print(model.translate(german_sentence))

for epoch in range(100):

    model.train()
    accum_loss = 0
    for batch_idx, batch in tqdm(enumerate(train_iterator)):
        optimizer.zero_grad()
        input = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(input, target)
        target = target[1:, :].reshape(-1)
        output = output[1:, ...].reshape(-1, output.shape[-1])
        loss = F.cross_entropy(output, target, ignore_index=pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        accum_loss += loss.item()

        if batch_idx % EVERY == 0 and batch_idx > 0:
            print(f'iteration {batch_idx}, average loss = {np.round(accum_loss/EVERY, 2)}')
            accum_loss = 0

    # Evaluation after each epoch:
    model.eval()
    print(model.translate(german_sentence))
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, f"./models/seq2seq_model_epoch{epoch}.pth.tar")
