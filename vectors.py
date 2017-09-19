import gensim

class WordVectors():
    def load_wordvectors(self):
        return gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

'''
w = WordVectors()
v = w.load_wordvectors()
print len(v["computer"].tolist())
'''
