import numpy as np

class Decoder():

	"""
	tags: list of possible states
	start_probs: the initial probabilities for each tag in a dictionary
	trans_probs: transition probabilities in dictionary where each tag is a 
				key corresponding to a dictionary of prev tag
				combos represented as "TAG TAG" with their 
				respective transition probabilities
	"""
	def __init__(self, tags, start_probs, trans_probs):
		self.tags = tags
		self.start_probs = start_probs
		self.trans_prob = trans_prob
		# Init dp matrix
		self.V = []
		self.max_prob = 0.0
		# Init state sequence to return
		self.state_output = []
		

	"""
	Viterbi Implementation	
	Input: seq - sequence of word vecs to decode
		   emis_probs - emission probabilities generated for the given sequence
	"""
	def decode(self, seq, emis_probs):
		# Check that we actually got the sequence
		seq_len = len(seq)
		if seq_len == 0:
			return []

		# Init tag deltas
		deltas = {}
		for state in self.tags:
			deltas[tag] = {'prob': self.start_probs[tag] * emis_probs[tag][seq[0]], 'prev': None}
		self.V.append(deltas)		

		# Run Viterbi over the length of the observation set
		for i in range(1, len(seq)):
			self.V.append({})
			for state in self.tags:
				# Get the max transition probability given the previous two states
				probs_to_max = []
				for prev_state_t1 in self.tags:
					for prev_state_t2 in self.tags:
						prev_states_tag = prev_state_t1 + ' ' + prev_state_t2
						probs_to_max = self.V[i-1][prev_state_t1]['prob'] * self.trans_probs[state][prev_states_tag] 
				max_trans_prob = max(probs_to_max)
				# Record the max transition prob in the dp matrix
				for prev_state_t1 in self.tags:
					for prev_state_t2 in self.tags:
						prev_states_tag = prev_state_t1 + ' ' + prev_state_t2	
						if self.V[i-1][prev_state]['prob'] * self.trans_probs[state][prev_states_tag] == max_trans_prob:
							max_prob = max_trans_prob * emis_probs[state][seq[i]]
							self.V[i][state] = {'prob': max_prob, 'prev_state': prev_state_t1} 
							break

		# Get the highest prob
		self.max_prob = max(value['prob'] for value in self.V[-1].values())

		# Find the beginning of the backtrace
		for state, data in self.V[-1].items():
			if data['prob'] == max_prob:
				self.state_output.append(state)
				prev = state
				break 	
		# Trace back to create the output state set
		for i in range(len(self.V)-2, -1, -1):
			self.state_output.insert(0, self.V[i+1][prev]['prev']
		
		return self.state_output


	"""
	Print the final set of decoded states
	"""
	def print_decoded_states(self):
		print 'Decoded States:' + ' '.join(self.state_output)
		print 'Max Probability:' + str(self.max_prob)


	"""
	Print the full table from Viterbi
	"""
	def print_dp_table(self):
		yield " ".join(("%12d" % i) for i in range(len(self.V)))
		for state in self.V[0]:
			yield "%.5s: " % state + " ".join("%.5s" % ("%f" % v[state]["prob"]) for v in self.V)
