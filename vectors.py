import gensim

class WordVectors():
    def load_wordvectors(self):
        return gensim.models.Word2Vec.load_word2vec_format('/data2/user_data/khite/ner_data/GoogleNews-vectors-negative300.bin', binary=True)

'''
w = WordVectors()
v = w.load_wordvectors()
print len(v["computer"].tolist())
'''
