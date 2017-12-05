"""
LSTM
"""

import tensorflow as tf

class ContextEmbeddings():

	"""
	input_sentences: List of lists that are each a sentence worth of word embeddings
	"""
	def __init__(self):
		# Create placeholder for input sentences (will be dynamically padded to max from input_sequences)
		self.word_embeddings = tf.placeholder(tf.int32, shape=[None, None])
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths") # (batch size, max length of sentence)	

	
	"""
	Create context vectors for input words based on adjacent words with bi-LSTMs
	input_sentences: List of lists that are each a sentence worth of word embeddings
	"""
	def embeddingSetup(self, input_sentences):
		# Define the input embeddings in tf
		L = tf.Variable(self.input_sequence, dtype=tf.float32, trainable=False)
		prior_embeddings = tf.nn.embedding_lookup(L, self.word_embeddings) # (batch, sentence, word_vector_size)

		# 1. get character embeddings
		K = tf.get_variable(name="char_embeddings", dtype=tf.float32, shape=[nchars, dim_char])
		# shape = (batch, sentence, word, dim of char embeddings)
		char_embeddings = tf.nn.embedding_lookup(K, char_ids)

		# 2. put the time dimension on axis=1 for dynamic_rnn
		s = tf.shape(char_embeddings) # store old shape
		# shape = (batch, x, sentence, word, dim of char embeddings)
		char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], s[-1]])
		word_lengths = tf.reshape(self.word_lengths, shape=[-1])

		# 3. bi lstm on chars
		cell_fw = tf.contrib.rnn.LSTMCell(char_hidden_size, state_is_tuple=True)
		cell_bw = tf.contrib.rnn.LSTMCell(char_hidden_size, state_is_tuple=True)

		_, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=word_lengths, dtype=tf.float32)
		# shape = (batch x sentence, 2 x char_hidden_size)
		output = tf.concat([output_fw, output_bw], axis=-1)

		# shape = (batch, sentence, 2 x char_hidden_size)
		char_rep = tf.reshape(output, shape=[-1, s[1], 2*char_hidden_size])

		# shape = (batch, sentence, 2 x char_hidden_size + word_vector_size)
		word_embeddings = tf.concat([pretrained_embeddings, char_rep], axis=-1)

		cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
		cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)

		(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
    						cell_bw, word_embeddings, sequence_length=sequence_lengths,
    						dtype=tf.float32)

		context_rep = tf.concat([output_fw, output_bw], axis=-1)


if __name__ == '__main__':
	ce = ContextEmbeddings()
