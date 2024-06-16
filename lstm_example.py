from TextGen.models import LSTM
from torch.utils import data
import matplotlib.pyplot as plt
import time
import re

class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y

def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1):
    num_train = int(len(sequences) * p_train)
    num_val = int(len(sequences) * p_val)

    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train + num_val]
    sequences_test = sequences[num_train + num_val:]

    def get_inputs_targets_from_sequences(sequences):
        inputs, targets = [], []
        for sequence in sequences:
            if len(sequence) > 3:
                inputs.append(sequence[:-1])
                targets.append(sequence[1:])
        return inputs, targets

    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

def plot_loss(training_loss, validation_loss):
    epochs = range(1, len(training_loss) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss, 'b', label='Training Loss')
    plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def custom_sentence_tokenize(text):
    sent_end_chars = ('.', '?', '!', ';')
    sentences = re.split(r'(?<=[{}])\s+'.format(''.join(sent_end_chars)), text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '' and '\n' not in sentence]
    return sentences

def custom_word_tokenize(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    return words

def gen_input(sequence):
    input_sequence = sequence[:-1]
    target_sequence = sequence[1:]
    return input_sequence, target_sequence

with open('data/StarWars-all.txt', 'r', encoding='utf-8') as file:
    text = file.read()
sentences = custom_sentence_tokenize(text)
words_list = []
for sentence in sentences:
    words = custom_word_tokenize(sentence)
    words_list.append(words)
training_set, validation_set, test_set = create_datasets(words_list, Dataset)

hidden_size=1
lstm = LSTM(hidden_size, 1000, words_list)

time_start = time.time()
lstm.train(training_set, validation_set, num_epochs=5)
time_end = time.time()

print(f'Training time: {time_end - time_start} seconds')

initial_sequence = ['qui', 'gon', 'i', 'think']
sequence_length = 30
num_generations = 5

for _ in range(num_generations):
    generated_text = lstm.generate_text(initial_sequence, sequence_length)
    print("Generated sequence:", ' '.join(generated_text))

perplexity = lstm.calculate_perplexity(training_set, 2207)
print(perplexity)
