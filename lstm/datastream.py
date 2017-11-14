import sys
import tensorflow as tf

from config import *
sys.path.insert(0, '../')

from data_util import Reader
from vectors import WordVectors

class DataStream():
    def __init__(self, config):
        self.config = config
        self.counter = 0
        r = Reader("eng.train")
        self.lines = r.process_words()
        w = WordVectors()
        self.vec = w.load_wordvectors()

    def has_next(self):
        return self.counter < 10 

    def next_data(self):
        labels = [self.config.tag_indices[word.tag] for word in self.lines[self.counter]]
        embeddings = [self.vec[word.word].tolist() if word.word in self.vec else [0] * self.config.context_size for word in self.lines[self.counter]]
        print "Labels {}".format(labels)
        self.counter += 1 
        tf_embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        tf_labels = tf.convert_to_tensor([labels], dtype=tf.float32)
        return tf_embeddings, tf_labels

    def reset(self):
        self.counter = 0
