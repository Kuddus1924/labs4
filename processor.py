import re

from num2words import num2words


class TextPreproc:
    def __init__(self, lemmatizer, stopwords, forbidden_chars=r'[^a-z0-9\s]'):
        self.__lemmatizer = lemmatizer
        self.__stopwords = set(stopwords)
        self.__forbidden_chars = re.compile(forbidden_chars)

    def __filter(self, string):
        return self.__forbidden_chars.sub(' ', string)

    def __lemmatize(self, words):
        return [self.__lemmatizer.lemmatize(w) for w in words]


    def __num2words(self, words):
        result = []
        for word in words:
            if word.isdigit():
                word = num2words(word)
                result.extend(self.__filter(word).split())
            else:
                result.append(word)
        return result

    def transform(self, sentences):
        return map(self.transform_sentence, sentences)

    def transform_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = self.__filter(sentence)

        words = sentence.split()
        words = self.__num2words(words)
        words = self.__lemmatize(words)
        words = self.__remove_stopwords(words)
        return ' '.join(words)

    def __remove_stopwords(self, words):
        return [w for w in words if w not in self.__stopwords]