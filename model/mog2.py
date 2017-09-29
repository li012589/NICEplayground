import numpy as np
import tensorflow as tf

class MixtureOfGaussians():
    def __init__(self, name='mog2'):
        self.z = tf.placeholder(tf.float32, [None, 2], name='z')

    def __call__(self, z):
        z1 = tf.reshape(tf.slice(z, [0, 0], [-1, 1]), [-1])
        z2 = tf.reshape(tf.slice(z, [0, 1], [-1, 1]), [-1])
        v1 = tf.sqrt((z1 - 5) * (z1 - 5) + z2 * z2) * 2
        v2 = tf.sqrt((z1 + 5) * (z1 + 5) + z2 * z2) * 2
        pdf1 = tf.exp(-0.5 * v1 * v1) / tf.sqrt(2 * np.pi * 0.25)
        pdf2 = tf.exp(-0.5 * v2 * v2) / tf.sqrt(2 * np.pi * 0.25)
        return -tf.log(0.5 * pdf1 + 0.5 * pdf2)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def std():
        return np.array([5.0, 0.5])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-8, 8]

    @staticmethod
    def ylim():
        return [-8, 8]
