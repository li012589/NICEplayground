import numpy as np
import tensorflow as tf

class doubleGaussian:
    '''
    A class represent doubleGaussian, when called return the energy function value.
    '''
    def __init__(self,name="doubleGaussian"):
        self.name = name
        self.z = tf.placeholder(tf.float32,[None,2])

    def __call__(self,z):
        with tf.variable_scope(self.name):
            z1 = tf.reshape(tf.slice(z,[0,0],[-1,1]),[-1])
            z2 = tf.reshape(tf.slice(z,[0,1],[-1,1]),[-1])
            v = tf.multiply(z1,z1)/2+tf.multiply(z2,z2)/2+tf.multiply(z1,z2)
            return v
    def mean(self,z):
        return np.mean(z,axis=1)
    def std(self,z):
        return np.std(z,axis=1)
    def measure(self,z,n):
        return np.mean(np.power(z,n),axis=1)

def main():
    '''
    Test script for doubleGaussian
    '''
    sess = tf.InteractiveSession()
    a = doubleGaussian("test")
    z = np.random.normal(0,1,[2,2])
    print(z)
    print(sess.run(a(z)))
    print(z[0][0])
    print(z[0][1])
    print(z[0][0]**2/2+z[0][1]**2/2+z[0][0]*z[0][1])
    print(a.mean(z))
if __name__ =="__main__":
    main()
