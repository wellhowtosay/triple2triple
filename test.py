import tensorflow as tf

samples = tf.random.categorical(tf.constant([[0.001, 0.5, 0.5, 0.6, 1],[0.001, 0.5, 0.5, 0.6, 1]]), 1)
print(samples)
print(samples[-1, 0])
