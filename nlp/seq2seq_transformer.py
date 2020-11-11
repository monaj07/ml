import numpy as np
import random
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


seed = 1364
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

spacy_eng = spacy.load('en')
spacy_ger = spacy.load('de')


def tokenizer_eng(text):
    return [token.text for token in spacy_eng.tokenizer(text)]


def tokenizer_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 device, nx=6, heads=8, pad_idx=1):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding_layer = PositionEmbedder(hidden_size, device=device,
                                                    pos_embedding_size=100)
        self.fc = nn.Linear(embedding_size, hidden_size)
        self.subencoder_list = nn.ModuleList(
            [SubEncoder(hidden_size, heads) for _ in range(nx)])
        self.pad_idx = pad_idx

    def forward(self, x):
        mask = (x != self.pad_idx)
        x_pos_embedded = self.pos_embedding_layer(x)
        x = self.embedding_layer(x)
        x = self.fc(x) + x_pos_embedded
        for _, layer in enumerate(self.subencoder_list):
            x = layer(x, mask)
        return x


class PositionEmbedder(nn.Module):
    def __init__(self, hidden_size, device, pos_embedding_size=100, dropout_p=0.5):
        super(PositionEmbedder, self).__init__()
        self.embedding_layer = nn.Embedding(pos_embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, seq):
        seq_len = seq.shape[0]
        batch_size = seq.shape[1]
        pos_raw = torch.arange(seq_len).unsqueeze(1).repeat(1, batch_size).to(device)
        pos_embedding = F.gelu(self.embedding_layer(pos_raw))
        pos_embedding = self.dropout(pos_embedding)
        return pos_embedding


class SubEncoder(nn.Module):
    def __init__(self, hidden_size, heads=8):
        super(SubEncoder, self).__init__()
        self.self_attention = Attention(hidden_size, heads)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask):
        att_out = self.self_attention(x, x, x, mask)
        middle_out = F.layer_norm(att_out + x, x.shape)
        fc_out = self.fc(middle_out)
        out = F.layer_norm(middle_out + fc_out, middle_out.shape)
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, heads=8, dropout_p=0.5):
        super(Attention, self).__init__()
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.heads = heads
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        seq_len = q.shape[0]
        batch_size = q.shape[1]
        feature_size = q.shape[2]
        head_dim = int(feature_size // self.heads)
        assert self.heads * head_dim == feature_size, "feature dimension is not divisible by heads"
        q_batch_first = q.permute(1, 0, 2)
        q_heads = q_batch_first.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        k_batch_first = k.permute(1, 0, 2)
        k_heads = k_batch_first.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 3, 1)
        v_batch_first = v.permute(1, 0, 2)
        v_heads = v_batch_first.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        energy = torch.matmul(q_heads, k_heads)
        energy_masked = energy * mask.permute(1, 0).view(batch_size, 1, 1, seq_len)
        energy_masked = energy_masked / np.sqrt(head_dim)
        att_heads = self.dropout(F.softmax(energy_masked, dim=-1))
        v_att = torch.matmul(att_heads, v_heads)
        v_att = v_att.permute(2, 0, 1, 3).reshape(seq_len, batch_size, -1)
        v_att = self.fc(v_att)
        return v_att


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    field_eng = Field(sequential=True, lower=True,
                     use_vocab=True, tokenize=tokenizer_eng,
                     init_token='<sos>', eos_token='<eos>')
    field_ger = Field(sequential=True, use_vocab=True,
                      init_token="<sos>", eos_token="<eos>",
                      lower=True, tokenize=tokenizer_ger)

    train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"),
                                                        fields=(field_ger, field_eng))
    field_eng.build_vocab(train_data, min_freq=2, max_size=10000)
    field_ger.build_vocab(train_data, min_freq=2, max_size=10000)

    train_iterator, valid_iterator, test_iterator = \
        BucketIterator.splits((train_data, valid_data, test_data),
                              batch_size=64, device=device,
                              sort_within_batch=True, sort_key=lambda x: len(x.src))

    n_epochs = 10
    embedding_size = 300
    hidden_size = 512
    heads = 8
    nx_encoder = 6
    eng_vocab_size = len(field_eng.vocab)
    ger_vocab_size = len(field_ger.vocab)

    encoder = Encoder(ger_vocab_size,
                      embedding_size,
                      hidden_size,
                      device,
                      nx=nx_encoder,
                      heads=heads,
                      pad_idx=field_ger.vocab.stoi['<pad>'])

    for epoch in range(n_epochs):
        for idx, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg
            output = encoder(src)
            print(src, trg)
            break