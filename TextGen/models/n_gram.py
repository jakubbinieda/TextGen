import math
from collections import defaultdict
import random


class NGram:
    def __init__(self, n=2) -> None:
        self.n = n
        self.model = defaultdict(lambda: defaultdict(int))

    def train(self, tokens: list[str]) -> None:
        for i in range(len(tokens) - self.n):
            ngram = tuple(tokens[i:i + self.n])
            next_word = tokens[i + self.n]
            self.model[ngram][next_word] += 1

    def generate(self, seed, length=100) -> list[str]:
        output = seed.split(' ')
        ngram = tuple(output[-self.n:])

        for _ in range(length):
            if ngram not in self.model:
                raise ValueError('Seed not in model')

            candidates = list(self.model[ngram].keys())
            candidates_probs = list(self.model[ngram].values())
            candidates_cnt = sum(candidates_probs)

            candidates_probs = [
                count / candidates_cnt for count in candidates_probs]

            next_word = random.choices(
                candidates,
                weights=candidates_probs,
                k=1)[0]

            output.append(next_word)
            ngram = tuple(output[-self.n:])

        return output

    def get_perplexity(self, tokens: list[str]) -> float:
        log_perplexity = 0
        N = len(tokens)
        for i in range(N - self.n):
            ngram = tuple(tokens[i:i + self.n])
            next_word = tokens[i + self.n]
            if ngram not in self.model or next_word not in self.model[ngram]:
                continue

            probability = self.model[ngram][next_word] / \
                sum(self.model[ngram].values())
            log_perplexity -= math.log(probability)

        return math.exp(log_perplexity / N)
