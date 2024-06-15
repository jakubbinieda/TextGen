import numpy as np
import torch
import torch.nn as nn

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")

best_model, word_to_int = torch.load("single-word.pth")
n_vocab = len(word_to_int)
int_to_word = dict((i, c) for c, i in word_to_int.items())


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


model = CharModel()
model.load_state_dict(best_model)

prompt = "the force is strong"
pattern = [word_to_int[c] for c in prompt.split(' ')]

model.to(device)

softmax = nn.Softmax()
model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(100):
        x = np.reshape(pattern, (1, len(pattern)))
        x = torch.tensor(x, dtype=torch.long)
        prediction = model(x.to(device))

        probs = softmax(prediction.cpu()[0])
        probs = np.array(probs)
        probs /= probs.sum()
        index = np.random.choice(len(probs), p=probs)
        # index = int(prediction.argmax())
        result = int_to_word[index]
        print(result, end=" ")
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")
