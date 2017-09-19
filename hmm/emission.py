import sys
import pickle
import os.path
import numpy as np
from sklearn.svm import SVC

sys.path.insert(0, '../')

from data_util import Reader
from vectors import WordVectors

class Emitter():
    def __init__(self, dataset):
        self.dataset = dataset
        self.symbol_indices = None
        self.models = None

    def convert_symbol_to_matrix(self, symbol):
        if self.symbol_indices is None:
            raise Exception("Model has not been trained")
        matrix = [0] * len(self.symbol_indices.keys())
        index = self.symbol_indices.get(symbol, -1)
        if index >= 0 and index < len(matrix):
            matrix[index] = 1
        return matrix

    def load_or_calculate(self):
        if os.path.isfile(self.dataset + ".model.pickle") and os.path.isfile(self.dataset + ".symbols.pickle"):
            self.models = pickle.load(open(self.dataset + ".model.pickle", "rb"))
            self.symbol_indices = pickle.load(open(self.dataset + "symbols.pickle", "wb"))
        else:
            self.train()

    def train(self):
        print "Getting training data"
        X_train, Y_train = self.generate_training_data()
        print "Fitting Models"
        count = 1
        self.models = []
        for key in self.symbol_indices.keys():
            print "Model: {} of {}".format(count, len(self.symbol_indices.keys()))
            Y_for_class = Y_train[self.symbol_indices[key]]
            model = SVC(probability=True)
            model = model.fit(X_train, Y_for_class)
            self.models.append(model)
            count += 1
        print "Successfully fit models"
        pickle.dump(self.models, open(self.dataset + ".model.pickle", "wb"))
        pickle.dump(self.symbol_indices, open(self.dataset + "symbols.pickle", "wb"))

    def generate_training_data(self):
        r = Reader(self.dataset)
        lines = r.process_words()
        self.enumerate_symbols(lines)

        w = WordVectors()
        vec = w.load_wordvectors()

        X = []
        Y = []
        for line in lines:
            for word in line:
                if word.word in vec:
                    X.append(vec[word.word].tolist())
                    Y.append(self.convert_symbol_to_matrix(word.tag))
                else:
                    X.append([0]*300)
                    Y.append(self.convert_symbol_to_matrix(word.tag))
        X = np.array(X)
        Y = np.array(Y).T
        return X, Y

    def enumerate_symbols(self, lines):
        tags = {}
        for line in lines:
            for word in line:
                tags[word.tag] = 1
        keys = tags.keys()
        self.symbol_indices = {keys[i]: i+1 for i in range(len(tags))}
        self.symbol_indices["<START>"] = 0

    def emit(self, word_data):
        if self.model is None:
            print "No model loaded, doing it now..."
            self.load_or_calculate()
        classes = self.model.predict_proba(word_data)[0]
        return {key: classes[self.symbol_indices[key]] for key in self.symbol_indices.keys()}

emitter = Emitter("eng.train")
emitter.train()
