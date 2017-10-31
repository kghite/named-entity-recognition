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
		self.trans_probs = trans_probs
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

		# Add the initial two wordvec predictions as the emission probs b/c no prior wordvecs to use
		initials = {}
		for state in self.tags:
			initials[state] = {'prob': self.start_probs[state] * emis_probs[state][0], 'prev_state': None}
		self.V.append(initials)		

		# Run Viterbi over the rest of the observation set
		for i in range(1, len(seq)):
			self.V.append({})
			for state in self.tags:
				# Get the prob of two prev states in sequence times the transition prob to the current state
				# for each set of prev states then maximize
				probs_given_prev = []
				for prev_tag in self.tags:
					if prev_tag in self.trans_probs[state]:
						t_prob = self.trans_probs[state][prev_tag]
					else:
						t_prob = 0.0
					probs_given_prev.append((self.V[i-1][prev_tag]['prob']) * t_prob) 
				max_trans_prob = max(probs_given_prev)
				# Figure out which tag pair had the max prob and then store in the dp matrix
				for prev_tag in self.tags:			
					if prev_tag in self.trans_probs[state]:
						t_prob = self.trans_probs[state][prev_tag]
					else:
						t_prob = 0.0
					tags_prob = ((self.V[i-1][prev_tag]['prob']) * t_prob)
					if tags_prob == max_trans_prob:
							max_prob = (max_trans_prob+0.2) * (emis_probs[state][i])
							self.V[i][state] = {'prob': max_prob, 'prev_state': prev_tag} 
							break

		# Get the highest prob
		self.max_prob = max(value['prob'] for value in self.V[-1].values())
		previous = None

		# Find the beginning of the backtrace
		for state, data in self.V[-1].items():
			if data['prob'] == self.max_prob:
				self.state_output.append(state)
				previous = state
				break 
	
		# Trace back to create the output state set
		for i in range(len(self.V)-2, -1, -1):
			self.state_output.insert(0, self.V[i+1][previous]['prev_state'])
			previous = self.V[i+1][previous]['prev_state']

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
