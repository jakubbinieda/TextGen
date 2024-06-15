import string
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")

filename = "data/StarWars-Episode1.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

translator = str.maketrans('', '', string.punctuation)
raw_text = raw_text.translate(translator)

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for c, i in char_to_int.items())

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length))
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(dataY, dtype=torch.long)


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x


n_epochs = 160
batch_size = 128
model = CharModel()
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(
    data.TensorDataset(
        X,
        y),
    shuffle=True,
    batch_size=batch_size)

best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    start_time = time.time()
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(
        "Epoch %d: Cross-entropy: %.4f, Time: %.2f seconds" %
        (epoch, loss, epoch_time))

torch.save([best_model, char_to_int], "single-char.pth")
