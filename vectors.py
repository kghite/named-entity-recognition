import gensim

print "Loading Model"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print "Model Loaded"
print word_vectors["queen"]
