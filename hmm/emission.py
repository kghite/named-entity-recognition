import pickle
import numpy as np
from sklearn.svm import SVC

class Emitter():
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None

    def convert_symbol_to_number(self, symbol):
        return 1 if symbol.lower() == self.symbol.lower() else 0

    def train(self, data, symbols):
        training_output = np.array([self.convert_symbol_to_number(symbol) for symbol in symbols])
        model = SVC(probability=True)
        self.model = model.fit(data, training_output)

    def emit(self, word_data):
        if self.model is not None:
            return self.model.predict_proba(word_data)[0][0]
        raise Exception("Model is not trained")

loc_emitter = Emitter("LOC")
training_input = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7], [3, 6, 7]])
training_output = ["LOC", "NONE", "NONE", "NONE"]

loc_emitter.train(training_input, training_output)
print loc_emitter.emit(np.array([[2, 3, 30]]))
