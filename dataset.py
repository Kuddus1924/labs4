import os

from glob import glob
from tqdm import tqdm


class Dataset:
    def __init__(self, preprocessor):
        self.__preprocessor = preprocessor
    def readdataTest(self, filename):
        return self.__preprocess(filename)
    def readdataTrain(self, glob_pattern):
        X = []
        Y = []
        for path in glob(glob_pattern):
            lines = self.__preprocess(path)
            X.extend(lines)
            y = self.__path(path)
            Y.extend([y] * len(lines))
        return X, Y
    def __path(self, path):
        filename = os.path.split(path)[-1]
        return int(filename[0])
    def __preprocess(self, path):
        processed_lines = []
        with open(path) as inf:
            lines = inf.readlines()
            transformer = self.__preprocessor.transform(lines)
            filename = os.path.split(path)[-1]
            for line in tqdm(transformer, desc=filename, total=len(lines)):
                processed_lines.append(line)
        return processed_lines