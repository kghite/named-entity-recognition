"""
Hidden Markov Model Implementation for Named Entity Recognition

Model Requirements:
	training set - 
	test sequence
"""

class Model():

	def __init__(self, training_set, start_probs):
		self.training_set = training_set
		self.start_probs = start_probs

		# Getting tags from transition probs
		self.tags = []
	
		# Create the transition and emission interfaces
		self.transition = Transition(self.training_set)
		emitter = Emitter(self.training_set)


	"""
	Decode data into wordvec and labels lists
	"""
	def generateSequenceData(self, input_sequence):
		pass


	"""
	Calculate accuracy of the decoded predictions compared to the tags
	"""
	def getAccuracy(self, wordvec_sequence, data_labels, decoded_labels):
		pass


	"""
	Get the probabilities and run the decoder
	"""
	def run(self, input_sequence):
		# Get transistion probs
		t_probs = transition.load_or_calculate()
	
		# Get emission probs
		# e_probs = self.getEmissionProbs(sequence)

		### DEBUG ###
		print "Transition probabilities"
		print t_probs 
		print "Emission probabilities"
		print e_probs

		# Run decoder 
		# decoder = Decoder(self.tags, self.start_probs, e_probs, t_probs)
		# decoded_states = decoder.decode(sequence)

		# decoder.print_decoded_states()
		# decoder.print_dp_table()
