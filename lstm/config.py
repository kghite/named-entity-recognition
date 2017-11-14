import tensorflow as tf
import os

class Config():
    def __init__(self):
        self.context_size = 300
        self.embeddings_size = 300
        self.ntags = 5
        self.learning_rate = .6
        self.tag_indices = {
            "O":0,
            "PER":1,
            "ORG":2,
            "LOC":3,
            "MISC":4
        }
        self.batch_size = 1
        self.number_of_epochs = 1 
        self.sess = tf.Session()

    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "model.ckpt")

    def load_model_if_exists(self):
        if os.path.isfile("model.ckpt.index"):
            saver = tf.train.Saver()
            saver.restore(self.sess, "model.ckpt")
            return True
        return False
