import os
import tensorflow as tf

from config import *

# Silence TF info logs
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class CRF():
    def __init__(self, config):
        self.config = config

    def run(self):
        hello = tf.constant("Hello, I am a CRF")
        print self.config.sess.run(hello)

if __name__ == "__main__":
    config = Config()
    crf = CRF(config)
    crf.run()
