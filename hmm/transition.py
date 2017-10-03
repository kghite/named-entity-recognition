import os.path
import sys
import pickle
import pprint

sys.path.insert(0, '../')
pp = pprint.PrettyPrinter(indent=2)	

from data_util import Reader 

class Transition():
    def __init__(self, dataset):
        self.dataset = dataset

    def load_or_calculate(self):
        if os.path.isfile(self.dataset + ".pickle"):
            return pickle.load(open(self.dataset + ".pickle", "rb"))
        r = Reader(self.dataset)
        words = r.process_words()
        transitions = self.calculate_transition_probability(words)
        pickle.dump(transitions, open(self.dataset + ".pickle", "wb"))
        return transitions

    def calculate_transition_probability(self, words):
        c_transitions = self.count_transitions(words)
        p_transitions = {} 
        for key in c_transitions:
            total = sum([value for i_key, value in c_transitions[key].iteritems()])
            probability_dictionary = {i_key: float(value)/float(total) for i_key, value in c_transitions[key].iteritems()}
            p_transitions[key] = probability_dictionary
        return p_transitions

    def count_transitions(self, words):
        transitions = {}
        for line in words:
            n_2 = "<START>"
            n_1 = "<START>"
            for full_word in line:
                n = full_word.tag
                n_transitions = transitions.get(n, {})
                bigram = "{} {}".format(n_2, n_1)
                transition_count = n_transitions.get(bigram, 0)
                n_transitions[bigram] = transition_count + 1
                transitions[n] = n_transitions
                n_2 = n_1
                n_1 = n

        return transitions


if __name__ == "__main__":
	t = Transition("eng.train")
	pp.pprint(t.load_or_calculate())
