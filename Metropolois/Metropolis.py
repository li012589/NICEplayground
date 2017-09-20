if __name__ =="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

import tensorflow as tf
from utils.expLogger import expLogger
from utils.MetropolisHastingsAccept import metropolisHastingsAccept
import numpy as np

class MHSampler():
    '''
    Tensorflow implementation for Metroplois-Hastings Monte Carlo sampler
    '''
    def __init__(self,energyFn,prior,std=1.0):
        self.energyFn = energyFn
        self.prior = prior
        self.z = self.energyFn.z
        expLog = expLogger({})
        def fn(z,x):
            z_ = z+tf.random_normal(tf.shape(self.z),0.0,std)
            accept = metropolisHastingsAccept(energyFn(z),energyFn(z_),expLog)
            return tf.where(accept,z_,z)
        self.steps = tf.placeholder(tf.int32,[])
        elems = tf.zeros([self.steps])
        self.z_ = tf.scan(fn,elems,self.z,back_prop=False)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    def sample(self,steps,batchSize):
        z = self.sess.run(self.z_,feed_dict={self.steps:steps,self.z:self.prior(batchSize)})
        #z = np.transpose(z,[1,0,2])
        return z
def main():
    from model.ring2d import Ring2d
    from model.doubleGaussian import doubleGaussian
    from model.phi4 import phi4
    from hmc.hmc import HMCSampler

    def prior(bs):
        return np.random.normal(0,1,[bs,2])
    energyFn = Ring2d()
    MH = MHSampler(energyFn,prior)
    HMC = HMCSampler(energyFn,prior)
    z_ = MH.sample(8000,800)
    z_ = z_[:,3000:]
    z_ = np.reshape(z_,[-1,2])
    x_,y_ = z_[:,0],z_[:,1]
    print("MH result")
    print(np.mean(x_))
    print(np.std(x_))
    z_ = HMC.sample(8000,800)
    z_ = z_[:,3000:]
    z_ = np.reshape(z_,[-1,2])
    x_,y_ = z_[:,0],z_[:,1]
    print("HMC result")
    print(np.mean(x_))
    print(np.std(x_))

if __name__ == "__main__":
    main()