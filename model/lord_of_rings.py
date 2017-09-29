import numpy as np
import tensorflow as tf

class LordOfRings():
    def __init__(self, name='lord_of_rings'):
        self.name = name
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
            z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
            v1 = (tf.sqrt(z1 * z1 + z2 * z2) - 1) / 0.2
            v2 = (tf.sqrt(z1 * z1 + z2 * z2) - 2) / 0.2
            v3 = (tf.sqrt(z1 * z1 + z2 * z2) - 3) / 0.2
            v4 = (tf.sqrt(z1 * z1 + z2 * z2) - 4) / 0.2
            v5 = (tf.sqrt(z1 * z1 + z2 * z2) - 5) / 0.2
            p1, p2, p3, p4, p5 = v1 * v1, v2 * v2, v3 * v3, v4 * v4, v5 * v5
            return tf.minimum(tf.minimum(tf.minimum(tf.minimum(p1, p2), p3), p4), p5)

    @staticmethod
    def mean():
        return np.array([3.6])

    @staticmethod
    def std():
        return np.array([1.24])

