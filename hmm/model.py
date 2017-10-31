"""
Hidden Markov Model Implementation for Named Entity Recognition

See HiddenMarkovModel.md for process and data formats
"""

# External imports
import numpy as np
import nltk
import pickle, pprint
import sys, os.path
from timeit import default_timer

# Internal imports
sys.path.insert(0, '../')
from data_util import Reader
from vectors import WordVectors
from transition import Transition
from emission import Emitter
from decoder_1back import Decoder

pp = pprint.PrettyPrinter(indent=2)


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
		return X, Y


	"""
	Convert plaintext into wordvec list
	"""
	def generateTextObservations(text):
		X = []	
		text_tokens = nltk.tokenize(text) 
		
		for word in text_tokens:
			X.append(vec[word])

		return X			
	

	"""
	Calulate the emission probs matrix for a given observation set
	Provide progress bar b/c this takes a while
	NOTE: Must run after tags have been pulled from t_probs
	"""
	def getEmissionProbs(self, observation_set):
		e_probs = {}

		# Add initial tag keys
		for tag in self.tags:
			e_probs[tag] = []

		# Calculate e_prob for each word vector and update progress bar every 2%
		total_words = len(observation_set)
		print str(total_words) + " words"
		# for i in range(0, total_words):
			# Get prob
			# probs = self.emitter.emit([observation_set[i]])
			# for tag in probs:
				# e_probs[tag].append(probs[tag])
			# Update status bar
			# percentage = int((float(i) / float(total_words))*100)
			# bars = percentage / 2
			# space = 50 - bars
			# print '\r' + str(percentage) + "% [" + "|"*bars + " "*space + "] " + str(i),
		
		probs = self.emitter.emit_batch(observation_set)

		for i in range(0, total_words):
			for tag in probs[i]:
				e_probs[tag].append(probs[i][tag])

		return e_probs


	"""
	Calculate accuracy of the decoded predictions compared to the tags
	"""
	def getAccuracy(self, test_labels, decoded_labels):
		correct = 0
		for i in range(0, len(test_labels)):
			if decoded_labels[i] == test_labels[i]:
				correct += 1
		return (float(correct)/float(len(test_labels))) * 100


	"""
	Create the confusion matrix for the decoded predictions compared to the tags
	confusion = {tag1: {tag1: 0, tag2: 0 ...},
		     tag2: {...}}
	confusion tracks the tag confusion counts and then converts to tag accuracies
	"""
	def getConfusionMatrix(self, test_labels, decoded_labels):
		# Init confusion dict
		confusion = {}
		tag_counts = {}
		for tag in self.tags:
			tag_counts[tag] = 0
		for tag in self.tags:
			confusion[tag] = tag_counts.copy()
		
		# Run through the output list to prevent running over tags multiple times
		for i in range(0, len(test_labels)):
			# Create a list of the confusion accuracies for each tag
			true_tag = test_labels[i]
			decoded_tag = decoded_labels[i]
			confusion[true_tag][decoded_tag] += 1

		# Convert the counts to accuracy percentages
		for tag in self.tags:
			# Count how many times the tag occurs in the test sequence
			true_count = test_labels.count(tag)
			if true_count != 0: # If there were no true instances, there won't be any counts
				for count_tag in self.tags:
					confusion[tag][count_tag] = float(confusion[tag][count_tag]) / float(true_count) 		
		
		return confusion


	"""
	Get the probabilities and run the decoder
	Input: input_sequence - test data or plaintext to decode
		   test - True if using tagged test data, False if decoding plaintext
				  Reports accuracy if using test data
	"""
	def run(self, input_sequence, test=True):
		# Get transistion probs and tags list
		print "\nGenerating transition probabilites"
		s = default_timer()
		t_probs = self.transition.load_or_calculate()
		self.tags = t_probs.keys()	
		e = default_timer()
		print "DONE: " + str(e-s) + "s\n"

		# Convert the input sequence to an observation set
		print "Converting input to observation set"
		s = default_timer()
		if test:
			observation_set, test_labels = self.generateTestObservations(input_sequence)
		else:
			observation_set = self.generateTextObservations(input_sequence)
		e = default_timer()
		print "DONE: " + str(e-s) + "s\n" 

		# Get emission probs
		print "Generating emission probabilities"
		s = default_timer()
		e_probs = self.getEmissionProbs(observation_set)
		e = default_timer()
		print "DONE: " + str(e-s) + "s\n"		

		# DEBUG: Probabilities
		# print "Transition probabilities"
		# pp.pprint(t_probs) 
		# print "Emission probabilities"
		# pp.pprint(e_probs)

		# Run decoder with even start probabilities
		start_probs = {}
		even_prob = 1.0 / len(self.tags)
		for tag in self.tags:
			start_probs[tag] = even_prob
		decoder = Decoder(self.tags, start_probs, t_probs)
		decoded_states = decoder.decode(observation_set, e_probs)

		# Report the accuracy of tests
		if test:
			print "Calculating test accuracy"
			acc = self.getAccuracy(test_labels, decoded_states)
			print str(acc) + "% accurate to test labels"
			print "Calculating confusion matrix"
			conf = self.getConfusionMatrix(test_labels, decoded_states)
			pp.pprint(conf)

		# DEBUG: DP Table
		# decoder.print_decoded_states()
		# table_gen = decoder.print_dp_table()
		# for i in table_gen:
			# print i



if __name__ == "__main__":
	print "Creating model ..."
	hmm = Model("eng.train")
	
	# DEBUG: Observations
	# print "Generating observation set ..."
	# X, Y = hmm.generateTestObservations("eng.testa")
	# print X[0]
	# print Y[0]

	print "Running model ..."
	hmm.run("eng.testbsmall")
