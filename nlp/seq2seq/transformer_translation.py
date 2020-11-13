import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')


def tokenize_eng(text):
    return [token.text for token in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]


english_field = Field(sequential=True, use_vocab=True,
                      init_token='<sos>', eos_token='<eos>',
                      tokenize=tokenize_eng, lower=True)
german_field = Field(sequential=True, use_vocab=True,
                     init_token='<sos>', eos_token='<eos>',
                     tokenize=tokenize_ger, lower=True)

train_data, valid_data, test_data = \
    Multi30k.splits(exts=(".de", ".en"), fields=(german_field, english_field))

german_field.build_vocab(train_data, max_size=10000, min_freq=2)
english_field.build_vocab(train_data, max_size=10000, min_freq=2)


class Translator(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_size,
                 num_encoder_layers, num_decoder_layers,
                 src_pad_idx, num_heads, forward_expansion,
                 dropout_p, max_len, device):
        super(Translator, self).__init__()
        self.src_word_embed = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_word_embed = nn.Embedding(trg_vocab_size, embedding_size)
        self.src_position_embed = nn.Embedding(max_len, embedding_size)
        self.trg_position_embed = nn.Embedding(max_len, embedding_size)
        self.max_len = max_len
        self.device = device
        self.dropout_p = dropout_p
        self.transformer = nn.Transformer(nhead=num_heads,
                                          d_model=embedding_size,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=forward_expansion,
                                          dropout=dropout_p)
        self.fc = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.T == self.src_pad_idx
        return src_mask

    def forward(self, src, trg):
        src_length, N = src.shape
        trg_length, N = trg.shape

        src_position = torch.arange(0, src_length).unsqueeze(1)\
            .expand(src_length, N).to(self.device)
        trg_position = torch.arange(0, trg_length).unsqueeze(1)\
            .expand(trg_length, N).to(self.device)

        src_full_embed = self.dropout(self.src_position_embed(src_position)
                                      + self.src_word_embed(src))
        trg_full_embed = self.dropout(self.trg_position_embed(trg_position)
                                      + self.trg_word_embed(trg))
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_length)\
            .to(self.device)

        out = self.transformer(src=src_full_embed,
                               tgt=trg_full_embed,
                               src_key_padding_mask=src_padding_mask,
                               tgt_mask=trg_mask)
        out = self.fc(out)
        return out


def translate_sentence(model, german_sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load("de")

    tokenised_german_sentence = [tok.text.lower() for tok in spacy_ger.tokenizer(german_sentence)]
    tokenised_german_sentence.insert(0, german_field.init_token)
    tokenised_german_sentence.append(german_field.eos_token)
    tokenised_german_indices = [german_field.vocab.stoi[token] for token in tokenised_german_sentence]
    sentence_tensor = torch.LongTensor(tokenised_german_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_epochs = 5
lr = 3e-4
batch_size = 32
src_vocab_size = len(german_field.vocab)
trg_vocab_size = len(english_field.vocab)
num_heads = 8
embedding_size = 512
num_encoder_layers = 3
num_decoder_layers = 3
dropout_p = 0.1
max_len = 100
forward_expansion = 4
src_padding_idx = german_field.vocab.stoi['<pad>']

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x:len(x.src),
    device=device
)

model = Translator(src_vocab_size, trg_vocab_size, embedding_size,
                   num_encoder_layers, num_decoder_layers,
                   src_padding_idx, num_heads, forward_expansion,
                   dropout_p, max_len, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
pad_idx = english_field.vocab.stoi['<pad']

for epoch in range(num_epochs):

    sentence = "Eine Frau rennt zu ihrem Haus."
    model.eval()

    translation = translate_sentence(
        model, sentence, german_field, english_field, device, max_length=50)

    print(f"Translated example sentence: \n {translation}")

    model.train()
    every = 100
    loss_avg = 0
    for idx, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        input_data = batch.src.to(device)
        target_data = batch.trg.to(device)
        output = model(input_data, target_data[:-1, :])
        target = target_data[1:, :].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = F.cross_entropy(output, target, ignore_index=pad_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        loss_avg += loss.item()

        if idx % every == 0 and idx > 0:
            print(f'iteration {idx}/{len(train_iterator)}: avg loss = {np.round(loss_avg/every, 2)}.')
            loss_avg = 0
