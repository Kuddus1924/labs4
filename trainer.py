import numpy as np
import textwrap

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

class Train:

    def __init__(self, classifier, params,jobs ,vectorizer=None ):
        sss = StratifiedShuffleSplit(
            test_size= 0.3,
            random_state= 0)

        self.__grid_cv = GridSearchCV(
            OneVsRestClassifier(classifier),
            self.__params(params),
            cv=sss,
             n_jobs = jobs)

        self.__vectorizer = vectorizer

    def __params(self, params):
        return {f'estimator__{k}': v for k, v in params.items()}

    def train(self, X, Y):
        X = self.__vectorize(X)
        self.__grid_cv.fit(X, Y)

    def predict(self, X, path=None):
        X = self.__vectorize(X)
        prediction = self.__grid_cv.best_estimator_.predict(X)
        if path:
            self.__save_file(path, prediction)
        return prediction


    def __save_file(self, path, prediction):
        header = textwrap.dedent(f'''\
            Best score: {self.__grid_cv.best_score_}
            Best estimator: {self.__grid_cv.best_estimator_}
            Best params: {self.__grid_cv.best_params_}''')
        np.savetxt(
            f'{path}_{self.__grid_cv.best_score_:.4f}',
            prediction,
            header=header,
            fmt='%d')

    def __vectorize(self, data):
        return self.__vectorizer.transform(data) if self.__vectorizer else data
