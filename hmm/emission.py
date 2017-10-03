import sys
import pickle
import os.path
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

sys.path.insert(0, '../')

from data_util import Reader
from vectors import WordVectors

class Emitter():
    def __init__(self, dataset):
        self.dataset = dataset
        self.symbol_indices = None
        self.model = None

    def convert_symbol_to_matrix(self, symbol):
        if self.symbol_indices is None:
            raise Exception("Model has not been trained")
        return self.symbol_indices.get(symbol, -1) + 1

    def load_or_calculate(self):
        if os.path.isfile(self.dataset + ".model.pickle") and os.path.isfile(self.dataset + ".symbols.pickle"):
            self.model = pickle.load(open(self.dataset + ".model.pickle", "rb"))
            self.symbol_indices = pickle.load(open(self.dataset + ".symbols.pickle", "rb"))
        else:
            self.train()

    def generate_or_load_training_data(self):
        if os.path.isfile(self.dataset + ".training.pickle") and os.path.isfile(self.dataset + ".symbols.pickle"):
            self.symbol_indices = pickle.load(open(self.dataset + ".symbols.pickle", "rb"))
            training_data = pickle.load(open(self.dataset + ".training.pickle", "rb"))
            return training_data["X"], training_data["Y"]
        else:
            X, Y = self.generate_training_data()
            training_data = {"X": X, "Y":Y}
            pickle.dump(training_data, open(self.dataset + ".training.pickle", "wb"))
            return X, Y

    def train(self):
        print "Getting training data"
        X_train, Y_train = self.generate_or_load_training_data()
        print "Fitting Model"
        n_estimators = 35
        model = OneVsRestClassifier(BaggingClassifier(SVC(probability=True, verbose=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=2))
        self.model = model.fit(X_train, Y_train)
        print "Successfully fit model"
        pickle.dump(self.model, open(self.dataset + ".model.pickle", "wb"))

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
        Y = np.array(Y)
        return X, Y

    def enumerate_symbols(self, lines):
        tags = {}
        for line in lines:
            for word in line:
                tags[word.tag] = 1
        keys = tags.keys()
        self.symbol_indices = {keys[i]: i for i in range(len(tags))}
        pickle.dump(self.symbol_indices, open(self.dataset + ".symbols.pickle", "wb"))

    def emit(self, word_data):
        if self.model is None:
            print "No model loaded, doing it now..."
            self.load_or_calculate()
        classes = self.model.predict_proba(word_data)[0]
        return {key: classes[self.symbol_indices[key]] for key in self.symbol_indices.keys()}


if __name__ == "__main__":        
	emitter = Emitter("eng.train")
	
	#emitter.train()
	
	w = WordVectors()
	vec = w.load_wordvectors()
	print emitter.emit([vec["Massachusetts"]])
