import numpy as np
import pickle
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, input_size, num_rnn_layers, hidden_size, output_size):
        super(RNN, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        embedding = self.embedding(x)
        out, (h, c) = self.lstm(embedding.unsqueeze(1), (h, c))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out, (h, c)

    def init_weights(self, batch_size):
        h = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size)
        return h, c


class Generation:
    def __init__(self, num_rnn_layers=2, hidden_size=256):
        self.all_characters = string.printable
        vocab_size = len(self.all_characters)
        self.rnn = RNN(vocab_size, num_rnn_layers, hidden_size, vocab_size).to(device)
        self.temperature = 0.85

    def char_tensor(self, text):
        output = [self.all_characters.index(c) for c in text]
        output = torch.tensor(output, dtype=torch.long)
        return output

    def train(self, length=250, iterations=5000, lr=0.003):
        with open('data/names.txt', 'r') as f:
            names = f.read()
        optimizer = optim.Adam(self.rnn.parameters(), lr=lr)

        loss_avg = 0
        self.rnn.train()
        for it in tqdm(range(iterations)):
            optimizer.zero_grad()
            start_idx = np.random.randint(0, len(names) - length - 1)
            data = names[start_idx: start_idx + length + 1]
            input = self.char_tensor(data[:-1]).to(device=device)
            output = self.char_tensor(data[1:]).to(device=device)
            h, c = self.rnn.init_weights(1)
            loss = 0
            for idx in range(length):
                if idx > 0 and it > 1000 and np.random.choice(1+(iterations - it)//500) == 0:
                    # Teacher forcing (Thanks to Sadegh Aliakbarian for introducing me to this idea)
                    out_wide = out.data.view(-1).div(self.temperature).exp()
                    out = torch.multinomial(out_wide, 1)
                    in_char = self.char_tensor(self.all_characters[out.item()]).to(device=device)
                else:
                    in_char = input[idx]
                out, (h, c) = self.rnn(in_char.unsqueeze(0), h, c)
                loss += F.cross_entropy(out, output[idx].unsqueeze(0))
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            if it % 50 == 0 and it > 0:
                print(f'average loss over the past 50 iterations: {np.round(loss_avg / 50, 2)}')
                loss_avg = 0
                print(self.generate())

    def generate(self, initial='M', length=50):
        self.rnn.eval()
        initial_input = self.char_tensor(initial).to(device=device)
        h, c = self.rnn.init_weights(1)
        for idx in range(len(initial) - 1):
            _, (h, c) = self.rnn(initial_input[idx].unsqueeze(0), h, c)

        prediction = list(initial)
        out = initial_input[-1].unsqueeze(0)
        for idx in range(len(initial) - 1, length):
            out, (h, c) = self.rnn(out, h, c)
            out_wide = out.data.view(-1).div(self.temperature).exp()
            out = torch.multinomial(out_wide, 1)
            prediction.append(self.all_characters[out.item()])
        return "".join(prediction)


generator = Generation()
generator.train()
print(generator.generate())
with open('models/text_generator.pkl', 'wb') as f:
    pickle.dump(generator, f)
