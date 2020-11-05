import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import spacy

seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


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


def tokenizer_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


def tokenizer_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_p):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=1)

    def forward(self, x):
        embedding = self.dropout(self.embedding_layer(x))
        output, h = self.rnn(embedding)
        return h


class Decoder(nn.Module):
    """
    Designed based on:
    https://github.com/bentrevett/pytorch-seq2seq/raw/d876a1dcacd7aeeeeeaff2c9b806d23116df048f/assets/seq2seq6.png
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_p):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers=1)
        self.fc = nn.Linear(embedding_size + 2 * hidden_size, vocab_size)

    def forward(self, x, h, h0):
        embedding = self.dropout(self.embedding_layer(x))
        gru_input = torch.cat([embedding, h0], dim=-1)
        gru_output, h = self.rnn(gru_input, h)
        linear_input = torch.cat([gru_output, embedding, h0], dim=-1)
        output = self.fc(linear_input)
        return output, h


class Seq2SeqGru(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqGru, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing=0.5):
        context = self.encoder(src)
        h = context
        x = trg[0].unsqueeze(0)
        output = []
        for idx in range(1, len(trg)):
            out_dec, h = self.decoder(x, h, context)
            output.append(out_dec)
            greedy_output = out_dec.argmax(-1)
            x = trg[idx].unsqueeze(0) if np.random.uniform(0, 1) < teacher_forcing else greedy_output
        output = torch.cat(output).to(self.device)
        return output


def translate(model, src, field_de, field_en, max_output_len=50):
    model.eval()
    device = next(model.parameters()).device
    init_token_de = field_de.vocab.stoi['<sos>']
    stop_token_de = field_de.vocab.stoi['<eos>']
    init_token_en = field_en.vocab.stoi['<sos>']
    stop_token_en = field_en.vocab.stoi['<eos>']
    if isinstance(src, str):
        src_list = tokenizer_de(src)
    elif isinstance(src, list):
        src_list = src
    else:
        raise NotImplementedError
    src_idx = [field_de.vocab.stoi[w] for w in src_list]
    src_idx = [init_token_de] + src_idx + [stop_token_de]
    src_idx = torch.tensor(src_idx).unsqueeze(1).to(device)
    context = model.encoder(src_idx)
    h = context
    x = torch.tensor([init_token_en]).unsqueeze(1).to(device)
    output = []
    for idx in range(1, max_output_len):
        out_dec, h = model.decoder(x, h, context)
        greedy_output = out_dec.argmax(-1)
        output.append(greedy_output.item())
        x = greedy_output
        if (x.cpu().squeeze() == torch.tensor([stop_token_en])).item():
            break
    translation = [field_en.vocab.itos[item] for item in output]
    model.train()
    return translation


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    field_en = Field(sequential=True, use_vocab=True,
                     init_token='<sos>', eos_token='<eos>',
                     lower=True, tokenize=tokenizer_en)
    field_de = Field(sequential=True, use_vocab=True,
                     init_token='<sos>', eos_token='<eos>',
                     lower=True, tokenize=tokenizer_de)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(field_de, field_en))
    # print(vars(train_data.examples[0]))
    vocab_de = field_de.build_vocab(train_data, max_size=10000, min_freq=2)
    vocab_en = field_en.build_vocab(train_data, max_size=10000, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=64, device=device,
        sort_key=lambda x: len(x.src), sort_within_batch=True)

    n_epochs = 10
    lr = 0.001
    dropout_p = 0.5
    embedding_size = 300
    hidden_size = 1024

    encoder = Encoder(len(field_de.vocab), embedding_size, hidden_size, dropout_p)
    decoder = Decoder(len(field_en.vocab), embedding_size, hidden_size, dropout_p)
    seq2seq_gru = Seq2SeqGru(encoder, decoder, device)
    seq2seq_gru = seq2seq_gru.to(device)


    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)


    seq2seq_gru.apply(init_weights)
    print(seq2seq_gru)

    german_sentence = ['eine', 'gruppe', 'von', 'männern', 'lädt', 'baumwolle', 'auf', 'einen', 'lastwagen']
    english_translation = translate(seq2seq_gru, german_sentence,
                                    field_de, field_en, max_output_len=50)
    print("Tranlation using the model so far:\n", english_translation)

    optimizer = optim.Adam(seq2seq_gru.parameters(), lr=lr)

    for epoch in range(n_epochs):
        seq2seq_gru.train()
        loss_agg = []
        for it, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg
            output = seq2seq_gru(src.to(device), trg.to(device),
                                 teacher_forcing=max(0.5 - epoch / n_epochs, 0.05))
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                                   trg[1:].reshape(-1),
                                   ignore_index=field_en.vocab.stoi['<pad>'])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq_gru.parameters(), max_norm=1)
            optimizer.step()
            loss_agg.append(loss.item())

        print("--------------------------------------")
        print(f"""Epoch {epoch}: Average training loss: 
                  {np.round(np.mean(loss_agg), 2)}""")

        # Validation:
        loss_agg = []
        seq2seq_gru.eval()
        for it, batch in enumerate(valid_iterator):
            src = batch.src
            trg = batch.trg
            output = seq2seq_gru(src.to(device), trg.to(device),
                                 teacher_forcing=0)
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                                   trg[1:].reshape(-1),
                                   ignore_index=field_en.vocab.stoi['<pad>'])
            loss_agg.append(loss.item())

        print(f"""Epoch {epoch}: Average validation loss: 
                  {np.round(np.mean(loss_agg), 2)}""")
        english_translation = translate(seq2seq_gru, german_sentence,
                                        field_de, field_en, max_output_len=50)
        print("Tranlation using the model so far:\n", english_translation)