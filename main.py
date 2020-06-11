

import nltk
from dataset import Dataset
from trainer import Train
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from processor import TextPreproc
from sklearn.svm import LinearSVC
from vectorMethods import CharCountVectorizer
from vectorMethods import TfIdf
from vectorMethods import Word2Vec


charCount =  True
tfidf = False
word2 = False
def main():
    nltk.download('wordnet')
    nltk.download('stopwords')

    tp = TextPreproc(WordNetLemmatizer(), stopwords.words('english'))
    dataset = Dataset(tp)
    train_X, train_Y = dataset.readdataTrain('data/train/*.txt')
    test_X = dataset.readdataTest('data/test.txt')
    classifier = LinearSVC(dual=False)
    params = {'C': [10 ** p for p in range(-2, 5)]}
    if charCount:
        ngram = 3
        vectorizer = CharCountVectorizer(train_X, ngram)
        trainer = Train(classifier, params, -1, vectorizer)
        trainer.train(train_X, train_Y)
        trainer.predict(test_X, f'data/prediction/ngrams/ngrams_{ngram}')
    if tfidf:
        vectorizer = TfIdf(train_X)
        trainer = Train(classifier, params,  -1 ,vectorizer)
        trainer.train(train_X, train_Y)
        trainer.predict(test_X, 'data/prediction/tfidf/tfidf')
    if word2:
        size = 300
        vectorizer = Word2Vec(train_X, size=size, window=3, workers=4)
        trainer = Train(classifier, params,  -1 , vectorizer)
        trainer.train(train_X, train_Y)
        trainer.predict(test_X, 'data/prediction/word2vec/word2vec')

if __name__ == '__main__':
    main()