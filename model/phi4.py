import numpy as np
import tensorflow as tf

class phi4:
    '''
    A class represent the phi4 model
    '''
    def __init__(self,n,l,d,name="phi4"):
        self.name = name
        self.hoppingTable = dict()
        self.z = tf.placeholder(tf.float32,[None,n])
        for i in range(n):
            LK = n
            y = i
            self.hoppingTable[i] = {}
            for j in reversed(range(d)):
                LK = int(LK/l)
                xk = int(y/LK)
                y = y-xk*LK
                if xk < l-1:
                    self.hoppingTable[i][j] = i + LK
                else:
                    self.hoppingTable[i][j] = i + LK*(1-l)
                if xk > 0:
                    self.hoppingTable[i][j+d] = i - LK
                else:
                    self.hoppingTable[i][j+d] = i-LK*(1-l)
    def __call__(self,z):
        with tf.variable_scope(self.name):
            pass

    def mean(self,z):
        pass
    def std(self,z):
        pass
    def measure(self,z):
        pass

if __name__ == "__main__":
    '''
    Test script
    '''
    pass