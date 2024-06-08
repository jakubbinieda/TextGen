from nltk import word_tokenize, sent_tokenize


class Data:
    def __init__(self, raw: str) -> None:
        self.raw = raw.lower()
        pass

    def word_tokenize(self) -> list[str]:
        # TODO: change this to a custom tokenizer
        return word_tokenize(self.raw)

    def sentence_tokenize(self) -> list[str]:
        sentences = sent_tokenize(self.raw)
        tokens = []
        for sentence in sentences:
            tokens.extend(word_tokenize(sentence))

        return tokens

    def get_probabilites(self):
        tokens = self.word_tokenize()
        probabilities = {}
        for i in range(len(tokens) - 1):
            word = tokens[i]
            next_word = tokens[i + 1]
            if word not in probabilities:
                probabilities[word] = {}
            if next_word not in probabilities[word]:
                probabilities[word][next_word] = 0
            probabilities[word][next_word] += 1
        for word in probabilities:
            total = sum(probabilities[word].values())
            for next_word in probabilities[word]:
                probabilities[word][next_word] /= total
        return probabilities


class DataLoader:
    @staticmethod
    def load(path) -> Data:
        with open(path, 'r') as file:
            data = file.read()
        return Data(data)
