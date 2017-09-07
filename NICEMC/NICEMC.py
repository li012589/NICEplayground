if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from utils.MetropolisHastingsAccept import metropolisHastingsAccept

class NiceNetworkOperator:
    def __init__(self,network,energyFn):
        self.network = lambda inputs,direction: tf.cond(direction,network.forward(inputs),network.backward(inputs))
        self.energyFn = energyFn
    def __call__(self,inputs,steps,vDim,ifMH):
        if ifMH:
            def fn(z):
                v = tf.random_normal([z.get_shape[0],vDim])
                z_,v_ = self.network([z,v],(tf.random_uniform([]) < 0.5))
                accept = metropolisHastingsAccept(self.energyFn(z,v))
                z_ = tf.where(accept,z_,z)
                return z_,v_
        else:
            def fn(z):
                v = tf.random_normal([z.get_shape[0],vDim])
                z_,v_ = self.network([z,v],True)
                return z_,v_
        elems = tf.zeros([steps])
        return tf.scan(fn,elems,inputs,back_prop=False)

class NICEMCSampler:
    def __init__(self,energyFn,prior,generator,discriminator,b,m):
        self.energyFn = energyFn
        self.prior = prior
        self.generator = generator
        self.discriminator = discriminator
        self.b = b
        self.m = m
    def sample(self,steps,batchSize):
        pass
    def train(self):
        pass

if __name__ == "__main__":
    pass