import tensorflow as tf

class Config():
    def __init__(self):
        self.context_size = 600
        self.embeddings_size = 300
        self.ntags = 5
        self.learning_rate = .6
        self.sess = tf.Session()
