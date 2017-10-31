import tensorflow as tf

from config import *

class DataStream():
    def __init__(self, config):
        self.config = config
        self.counter = 0
        self.stop = 10

    def has_next(self):
        return self.counter < self.stop 

    def next_data(self):
        self.counter += 1
        fake_context_reps = tf.random_normal(
                shape=[1, self.config.context_size, 5],
                mean=10,
                stddev=5,
                dtype=tf.float32)
        fake_labels = tf.random_normal(
                shape=[1, 5],
                mean = (self.config.ntags/2),
                stddev=1,
                dtype=tf.float32)
        return fake_context_reps, fake_labels

    def reset(self):
        self.counter = 0
