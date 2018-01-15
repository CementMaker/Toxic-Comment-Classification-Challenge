import multiprocessing

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec

sentences = LineSentence("wiki.en.text")
print("loading sentences end")
model = Word2Vec(sentences, size=400, window=5, min_count=5, workers=8)
model.save("./model/word2vec.model")

