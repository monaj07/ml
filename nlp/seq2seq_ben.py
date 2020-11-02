import numpy as np
import random
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k
from tqdm import tqdm

# Setting a fixed seed for reproducibility
seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Setting spacy tokenizer languages
# python -m spacy download en
# python -m spacy download de
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')


def bleu(test_data, model, field_de, field_en):
    targets = []
    outputs = []

    for example in test_data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate(model, src, field_de, field_en, max_output_len=50)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


# def bleu(model, field_de, field_en):
#     targets = []
#     outputs = []
#     src = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen".split(" ")
#     trg = "a boat with several men on it is being pulled ashore by a large team of horses".split(" ")
#     # prediction = translate(model, src, field_de, field_en, max_output_len=50)
#     prediction = ['a', 'boat', 'with', 'several', 'men', 'begins', 'to', 'a', 'a', 'large', 'large', 'large', 'large', '<unk>', '<eos>']
#     prediction = prediction[:-1]  # remove <eos> token
#     targets.append([trg])
#     outputs.append(prediction)
#     return bleu_score(outputs, targets)


def translate(seq2seq, german_sentence, field_en, field_de, max_output_len=50):
    seq2seq.eval()
    encoder = seq2seq.encoder
    decoder = seq2seq.decoder
    device = next(seq2seq.parameters()).device
    # Pre-processing the example input german sentence:
    if isinstance(german_sentence, str):
        german_sentence_tokenized = [token.text for token in spacy_de.tokenizer(german_sentence)]
    elif isinstance(german_sentence, list):
        german_sentence_tokenized = german_sentence
    else:
        raise NotImplementedError
    german_sentence_tokenized.insert(0, field_de.init_token)
    german_sentence_tokenized.append(field_de.eos_token)
    german_sentence_indices = [field_de.vocab.stoi[token] for token in german_sentence_tokenized]
    german_sentence_tensor = torch.LongTensor(german_sentence_indices).unsqueeze(1).to(device)
    # German sentence  is pre-processed now fo feed the model,
    # From now on it quite resembles the decode method, but with teacher_forcing set to off:
    predictions = []
    vocab_init = field_en.vocab.stoi[field_en.init_token]
    vocab_stop = field_en.vocab.stoi[field_en.eos_token]
    h, c = encoder(german_sentence_tensor)
    trg = torch.Tensor([vocab_init]).unsqueeze(0).long().to(device)
    x = trg[0, ...].unsqueeze(0)
    for idx in range(1, max_output_len):
        if x.squeeze() == torch.Tensor([vocab_stop]).to(device):
            break
        out, h, c = decoder(x, h, c)
        # out_probs = F.softmax(out, dim=-1)
        x = out.argmax(dim=-1)
        predictions.append(x)

    translation = [field_en.vocab.itos[idx] for idx in predictions]
    seq2seq.train()
    return translation


def tokenizer_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


def tokenizer_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_embedding_size=300,
                 num_rnn_layers=2,
                 hidden_size=512,
                 dropout_p=0.5):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(word_embedding_size, hidden_size=hidden_size,
                           num_layers=num_rnn_layers, batch_first=False,
                           dropout=dropout_p, bidirectional=False)

    def forward(self, x):
        # x : [src_len, batch_size]
        embedded_input = self.dropout(self.embedding_layer(x))
        # embedded_input : [src_len, batch_size, word_embedding_size]
        outputs, (h, c) = self.rnn(embedded_input)
        # outputs : [src_len, batch_size, hidden_size * n_directions]
        # h, c : [num_rnn_layers * n_directions, batch_size, hidden_size]
        # outputs are always from the top hidden layer
        return h, c


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_embedding_size=300,
                 num_rnn_layers=2,
                 hidden_size=512,
                 dropout_p=0.5):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(word_embedding_size, hidden_size,
                           num_rnn_layers, batch_first=False,
                           bidirectional=False, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c):
        # x : [1, batch_size]
        embedding_input = self.dropout(self.embedding_layer(x))
        # embedding_input : [1, batch_size, word_embedding_size]
        out, (h, c) = self.rnn(embedding_input, (h, c))
        out = self.fc(out)
        return out, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing=0.5):
        predictions = []
        h, c = self.encoder(src)
        x = trg[0, ...].unsqueeze(0)
        for idx in range(1, trg.shape[0]):
            out, h, c = self.decoder(x, h, c)
            # out_probs = F.softmax(out, dim=-1)
            predictions.append(out)
            chance = np.random.uniform(low=0, high=1)
            x = trg[idx, ...].unsqueeze(0) if chance < teacher_forcing else out.argmax(dim=-1)

        predictions = torch.cat(predictions, dim=0)
        return predictions


if __name__ == '__main__':
    field_de = Field(sequential=True, use_vocab=True,
                     init_token='<sos>', eos_token='<eos>',
                     tokenize=tokenizer_de, lower=True)
    field_en = Field(sequential=True, use_vocab=True,
                     init_token='<sos>', eos_token='<eos>',
                     tokenize=tokenizer_en, lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(field_de, field_en))

    vocab_de = field_de.build_vocab(train_data, max_size=10000, min_freq=2)
    vocab_en = field_en.build_vocab(train_data, max_size=10000, min_freq=2)
    print(f'English vocabulary size: {len(field_en.vocab)}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=64, device=device,
        sort_within_batch=True, sort_key=lambda x: len(x.src))

    german_vocab_size = len(field_de.vocab)
    english_vocab_size = len(field_en.vocab)
    word_embedding_size = 300
    num_rnn_layers = 2
    hidden_size = 1024
    dropout_p = 0.5
    n_epochs = 100
    lr = 0.001

    encoder = Encoder(german_vocab_size,
                      word_embedding_size,
                      num_rnn_layers,
                      hidden_size,
                      dropout_p).to(device)
    decoder = Decoder(english_vocab_size,
                      word_embedding_size,
                      num_rnn_layers,
                      hidden_size,
                      dropout_p).to(device)
    seq2seq = Seq2Seq(encoder, decoder).to(device)

    german_sentence = "Mehrere Leute gehen am Strand spazieren und spielen Fußball."
    print("Tranlation using the model so far:")
    english_translation = translate(seq2seq, german_sentence,
                                    field_en, field_de,
                                    max_output_len=50)

    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    score = bleu(test_data[1:100], seq2seq, field_de, field_en)
    print(f"Bleu score {score * 100:.2f}")

    for epoch in range(n_epochs):
        loss_agg = []
        seq2seq.train()
        for it, batch in enumerate(train_iterator):
            input_seq, target_seq = batch.src, batch.trg
            output = seq2seq(input_seq.to(device), target_seq.to(device),
                             teacher_forcing=max(0.5 - epoch / n_epochs, 0.05))
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                                   target_seq[1:].reshape(-1),
                                   ignore_index=field_en.vocab.stoi[field_en.pad_token])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=1)
            optimizer.step()
            loss_agg.append(loss.item())

        print("--------------------------------------")
        print(f"""Epoch {epoch}: Average training loss: 
                  {np.round(np.mean(loss_agg), 2)}""")
        loss_agg = []

        # Evaluate the trained model so far on the validation set
        seq2seq.eval()
        loss_agg_val = []
        with torch.no_grad():
            for it, batch in enumerate(valid_iterator):
                input_seq, target_seq = batch.src, batch.trg
                output = seq2seq(input_seq.to(device), target_seq.to(device),
                                 teacher_forcing=0)
                loss = F.cross_entropy(output.view(-1, english_vocab_size),
                                       target_seq[1:].view(-1),
                                       ignore_index=field_en.vocab.stoi[field_en.pad_token])
                loss_agg_val.append(loss.item())
            print(f"""Epoch {epoch}: Average validation loss: 
                      {np.round(sum(loss_agg_val) / len(loss_agg_val), 2)}""")
            print("Tranlation using the model so far:")
            english_translation = translate(seq2seq, german_sentence,
                                            field_en, field_de,
                                            max_output_len=50)
            print(english_translation)
            print("--------------------------------------")
