import numpy as np

from gensim.models import Word2Vec as GensimWord2Vec
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer


class TfIdf:
    def __init__(self,line):
        self.__vectorizer = SklearnTfidfVectorizer()
        self.__vectorizer.fit(line)

    def transform(self,line):
        return self.__vectorizer.transform(line).toarray()

class CharCountVectorizer:
    def __init__(self, lines, ngram):
        self.__vectorizer = SklearnCountVectorizer(
            analyzer='char_wb',
            ngram_range=(ngram, ngram))
        self.__vectorizer.fit(lines)

    def transform(self, lines):
        return self.__vectorizer.transform(lines).toarray()

class Word2Vec:
    def __init__(self, lines, **kwargs):
        lines = [line.split() for line in lines]
        self.__model = GensimWord2Vec(lines, **kwargs)

    def transform(self, lines):
        lines = [line.split() for line in lines]
        return [self.__transform_line(line) for line in lines]

    def __transform_line(self, line):
        encoded_line = []
        for word in line:
            if word in self.__model.wv:
                encoded_line.append(self.__model.wv[word])
        return np.mean(encoded_line or [np.zeros(self.__model.vector_size)], axis=0)
