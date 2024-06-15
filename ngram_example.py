from TextGen.models import NGram
from TextGen.utils.data_loader import DataLoader
from TextGen.models import NGram

ngram = NGram(n=3)
data = DataLoader.load('data/StarWars-all.txt')
tokens = data.sentence_tokenize()
ngram.train(tokens)
new_tokens = ngram.generate('the force is strong', length=100)
print(' '.join(new_tokens))
