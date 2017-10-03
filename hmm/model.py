"""
Hidden Markov Model Implementation for Named Entity Recognition

Model Requirements:
	training set - 
	test sequence
"""

# External imports
import numpy as np
import nltk
import pickle
import sys, os.path

sys.path.insert(0, '../')
from data_util import Reader
from vectors import WordVectors

# Internal imports
from transition import Transition
from emission import Emitter
# import decoder



class Model():


	def __init__(self, training_set):
		self.training_set = training_set

		# Getting tags from transition probs
		self.tags = []
	
		# Create the transition and emission interfaces
		self.transition = Transition(self.training_set)
		self.emitter = Emitter(self.training_set)


	"""
	Convert test data into wordvec and labels lists
	Returns: X - numpy array of word vectors
			 Y - numpy array of tags matching each X vectors
	"""
	def generateTestObservations(self, input_sequence):
		r = Reader(input_sequence)
		lines = r.process_words()

		w = WordVectors()
		vec = w.load_wordvectors()

		X = []
		Y = []
		for line in lines:
			for word in line:
				if word.word in vec:
					X.append(vec[word.word].tolist())
					Y.append(word.tag)
				else:
					X.append([0]*300)
					Y.append(word.tag)
		X = np.array(X)
		Y = np.array(Y)
		return X, Y


	"""
	Convert plaintext into wordvec list
	"""
	def generateTextObservations(text):
		X = []	
		text_tokens = nltk.tokenize(text) 
		
		for word in text_tokens:
			X.append(vec[word])

		return np.array(X)			
	

	"""
	Calulate the emission probs matrix for a given observation set
	"""
	def getEmissionProbs(self, sequence):
		pass


	"""
	Calculate accuracy of the decoded predictions compared to the tags
	"""
	def getAccuracy(self, test_labels, decoded_labels):
		pass


	"""
	Get the probabilities and run the decoder
	Input: input_sequence - test data or plaintext to decode
		   test - True if using tagged test data, default False
				  Reports accuracy if using test data
	"""
	def run(self, input_sequence, test=True):
		# Get transistion probs
		t_probs = self.transition.load_or_calculate()
	
		# Convert the input sequence to an observation set
		if test:
			observation_set, test_labels = generateTestObservations(input_sequence)
		else:
			observation_set = generateTextObservations(input_sequence)

		# Get emission probs
		e_probs = self.getEmissionProbs(sequence)

		# DEBUG: Probabilities
		# print "Transition probabilities"
		# print t_probs 
		# print "Emission probabilities"
		# print e_probs

		# Run decoder 
		# decoder = Decoder(self.tags, start_probs, e_probs, t_probs)
		# decoded_states = decoder.decode(sequence)

		if test:
			getAccuracy(test_labels, decoded_states)

		# decoder.print_decoded_states()
		# decoder.print_dp_table()



if __name__ == "__main__":
	print "Creating model ..."
	hmm = Model("eng.train")
	
	# DEBUG: Observations
	# print "Generating observation set ..."
	# X, Y = hmm.generateTestObservations("eng.testa")
	# print X[0]
	# print Y[0]

	print "Running model ..."
	hmm.run("eng.testa")
