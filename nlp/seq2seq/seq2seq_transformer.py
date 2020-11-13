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
    def __init__(self, vocab_size, hidden_size, ff_hidden_size, device,
                 dropout_p=0.5, nx=6, heads=8, max_length=100):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding_layer = nn.Embedding(max_length, hidden_size)
        self.subencoder_list = nn.ModuleList(
            [SubEncoder(hidden_size, ff_hidden_size, heads) for _ in range(nx)])
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, src_mask):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        pos_raw = torch.arange(seq_len).unsqueeze(1).repeat(1, batch_size).to(self.device)
        x_pos_embedded = self.pos_embedding_layer(pos_raw)
        x = self.embedding_layer(x) * np.sqrt(self.hidden_size)
        x = self.dropout(x + x_pos_embedded)
        for layer in self.subencoder_list:
            x = layer(x, src_mask)
        return x


class SubEncoder(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size=2048, heads=8, dropout_p=0.5):
        super(SubEncoder, self).__init__()
        self.self_attention = Attention(hidden_size, heads)
        self.fc = PositionwiseFeedforwardLayer(hidden_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, src_mask):
        att_out, _ = self.self_attention(x, x, x, src_mask)
        middle_out = F.layer_norm(self.dropout(att_out) + x, x.shape)
        fc_out = self.fc(middle_out)
        out = F.layer_norm(middle_out + self.dropout(fc_out), middle_out.shape)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, ff_hidden_size, device,
                 dropout_p=0.5, nx=6, heads=8, max_length=100):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding_layer = nn.Embedding(max_length, hidden_size)
        self.subdecoder_list = nn.ModuleList(
            [SubDecoder(hidden_size, ff_hidden_size, heads) for _ in range(nx)])
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_size = hidden_size
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.device = device

    def forward(self, x, z, src_mask, trg_mask):
        trg_len = x.shape[0]
        batch_size = x.shape[1]
        pos_raw = torch.arange(trg_len).unsqueeze(1).repeat(1, batch_size).to(self.device)
        x_pos_embedded = self.pos_embedding_layer(pos_raw)
        x = self.embedding_layer(x) * np.sqrt(self.hidden_size)
        x = self.dropout(x + x_pos_embedded)
        for layer in self.subencoder_list:
            x, attention = layer(x, z, src_mask, trg_mask)
        out = self.fc_out(x)
        return out, attention


class SubDecoder(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size=2048, heads=8, dropout_p=0.5):
        super(SubDecoder, self).__init__()
        self.self_masked_attention = Attention(hidden_size, heads=heads)
        self.encoder_attention = Attention(hidden_size, heads=8)
        self.fc = PositionwiseFeedforwardLayer(hidden_size, ff_hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, z, src_mask, trg_mask):
        att_out, _ = self.self_masked_attention(x, x, x, trg_mask)
        middle_out_1 = F.layer_norm(self.dropout(att_out) + x, x.shape)
        att_out_2, _ = self.encoder_attention(middle_out_1, z, z, src_mask)
        middle_out_2 = F.layer_norm(self.dropout(att_out_2) + middle_out_1, middle_out_1.shape)
        fc_out = self.fc(middle_out_2)
        out = F.layer_norm(middle_out_2 + self.dropout(fc_out), middle_out_2.shape)
        return out


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout_p=0.5):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, heads=8, dropout_p=0.5):
        super(Attention, self).__init__()
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.heads = heads
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        batch_size = k.shape[1]
        feature_size = k.shape[2]
        head_dim = int(feature_size // self.heads)
        assert self.heads * head_dim == feature_size, "feature dimension is not divisible by heads"
        q_batch_first = q.permute(1, 0, 2)
        q_heads = q_batch_first.view(batch_size, -1, self.heads, feature_size).permute(0, 2, 1, 3)
        k_batch_first = k.permute(1, 0, 2)
        k_heads = k_batch_first.view(batch_size, -1, self.heads, feature_size).permute(0, 2, 3, 1)
        v_batch_first = v.permute(1, 0, 2)
        v_heads = v_batch_first.view(batch_size, -1, self.heads, feature_size).permute(0, 2, 1, 3)
        energy = torch.matmul(q_heads, k_heads) / np.sqrt(head_dim)
        if mask is not None:
            mask = mask.view(batch_size, 1, -1, head_dim).repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(mask == 0, -1e10)
        att_heads = F.softmax(energy, dim=-1)
        v_att = torch.matmul(self.dropout(att_heads), v_heads)
        v_att = v_att.permute(2, 0, 1, 3).reshape(-1, batch_size, feature_size)
        v_att = self.fc(v_att)
        return v_att, att_heads


class Seq2Seq_transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def forward(self, src, trg):
        src_len = src.shape[0]
        trg_len = trg.shape[0]
        batch_size = src.shape[1]
        src_mask = (src != self.src_pad_idx).permute(1, 0).\
            view(batch_size, src_len, 1).repeat(1, 1, src_len)
        trg_pad_mask = (trg != self.trg_pad_idx).permute(1, 0).\
            view(batch_size, trg_len, 1).repeat(1, 1, trg_len)
        trg_mask_no_future = torch.tril(torch.ones(trg_len, trg_len)).\
            view(1, trg_len, trg_len).repeat(batch_size, 1, 1).to(self.device)
        trg_mask = trg_pad_mask * trg_mask_no_future
        z = self.encoder(src, src_mask)
        out, attention = self.decoder(trg, z, src_mask, trg_mask)
        return out, attention


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
    hidden_size = 512
    ff_hidden_size = 2048
    heads = 8
    nx_encoder = 6
    eng_vocab_size = len(field_eng.vocab)
    ger_vocab_size = len(field_ger.vocab)

    encoder = Encoder(ger_vocab_size,
                      hidden_size,
                      ff_hidden_size,
                      device,
                      nx=nx_encoder,
                      heads=heads)

    for epoch in range(n_epochs):
        for idx, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg
            output = encoder(src)
            print(src, trg)
            break