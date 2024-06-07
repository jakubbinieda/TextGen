from nltk import word_tokenize


class Data:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        pass

    def word_tokenize(self) -> list[str]:
        # TODO: change this to a custom tokenizer
        return word_tokenize(self.raw)


class DataLoader:
    @staticmethod
    def load(path) -> Data:
        with open(path, 'r') as file:
            data = file.read()
        return Data(data)
