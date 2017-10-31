import os
import tensorflow as tf

from config import *
from datastream import *

# Silence TF info logs
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class CRF():
    def __init__(self, config):
        self.config = config
    
    def load_or_build(self):
        pass

    def predict(self):
        pass

    def train(self, data_stream):
        pass
      
    def initialize_vars(self):
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self.config.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        print "UNINITIALIZED {}".format(uninit_vars)
        self.config.sess.run(tf.variables_initializer(uninit_vars))
        #self.config.sess.run(tf.global_variables_initializer())

    def run_epoch(self, data_stream):
            while data_stream.has_next():
                print "ITERATION"
                ctx, lbl = data_stream.next_data()
                try:
                    training_op = self.get_training_op(ctx, lbl)
                    self.config.sess.run(training_op)
                except tf.errors.FailedPreconditionError:
                    self.initialize_vars()
                    training_op = self.get_training_op(ctx, lbl)
                    self.config.sess.run(training_op)

            data_stream.reset()

    def get_training_op(self, context_reps, labels):
            try:
                loss = self.get_loss_function(context_reps, labels)
            except ValueError:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    loss = self.get_loss_function(context_reps, labels)
            try:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
                minimize = optimizer.minimize(loss)
            except ValueError:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
                    minimize = optimizer.minimize(loss)
            return minimize

    def get_loss_function(self, context_reps, labels):
        scores = self.get_scoring_function(context_reps)
        labels = tf.cast(labels, dtype=tf.int32)
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(scores, labels, tf.constant([5]))
        self.trans_params = trans_params
        return tf.reduce_mean(-log_likelihood)
    
    def get_scoring_function(self, context_reps):
        W = tf.get_variable("W", dtype=tf.float32,
                shape=[self.config.context_size, self.config.ntags])
        b = tf.get_variable("b", dtype=tf.float32,
                shape=[self.config.ntags], initializer=tf.zeros_initializer())
        output = tf.reshape(context_reps, [-1, self.config.context_size])
        pred = tf.matmul(output, W) + b
        return tf.reshape(pred, [1, pred.get_shape()[0].value, pred.get_shape()[1].value])

    def run(self):
        hello = tf.constant("Hello, I am a CRF")
        print self.config.sess.run(hello)

if __name__ == "__main__":
    config = Config()
    crf = CRF(config)
    ds = DataStream(config)
    crf.run_epoch(ds)
