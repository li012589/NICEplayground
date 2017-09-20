if __name__ =="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

import tensorflow as tf
from utils.expLogger import expLogger
from utils.MetropolisHastingsAccept import metropolisHastingsAccept
from utils.hamiltonian import hamiltonian
import numpy as np

def simulateDynameics(initialPos,initialV,stepSize,steps,energyFn):
    """
    Run Hamilitonian evolution for some step
    :param intialPos: the states of field before evolution
    :param intiialV: the velocity before evolution
    :param stepSize: the size of a single step
    :param steps: total steps in the evolution
    :param energyFn: the energy fucntion used in the evolution
    :return newPos: the states of the field after evolution
    :return newV: the velocity after evolution
    """
    force = tf.gradients(tf.reduce_sum(energyFn(initialPos)),initialPos)[0]
    tmpV = initialV - 0.5*stepSize*force
    tmpPos = initialPos +stepSize*tmpV
    i = tf.constant(0)
    def condition(pos, vel, step, i):
        return tf.less(i, steps)
    def leapfrogMeta(pos,vel,step,i):
        force = tf.gradients(tf.reduce_sum(energyFn(pos)),pos)[0]
        newV = vel - step*force
        newPos = pos +step*newV
        i += 1
        return [newPos,newV,step,i]
    newPos,newV,_,_ = tf.while_loop(condition,leapfrogMeta,[tmpPos,tmpV,stepSize,i])
    force = tf.gradients(tf.reduce_sum(energyFn(newPos)),newPos)[0]
    newV -= 0.5*stepSize*force
    #newPos += stepSize*newV
    return newPos,newV

def hmcMove(initialPos,energyFn,stepSize,steps):
    """
    Perform a HMC move
    :param intialPos: the states of field before evolution
    :param energyFn: the energy fucntion used in the evolution
    :param stepSize: the size of a single step
    :param steps: total steps in the evolution
    :return accept: a bool variable of if the udpate is accepted
    :return newPos: the states of the field after evolution
    :return newV: the velocity after evolution
    """
    initialV = tf.random_normal(tf.shape(initialPos))
    newPos,newV = simulateDynameics(initialPos,initialV,stepSize,steps,energyFn)
    expLog = expLogger({})
    accept = metropolisHastingsAccept(hamiltonian(initialPos,initialV,energyFn),hamiltonian(newPos,newV,energyFn),expLog)
    return accept,newPos,newV

def hmcUpdate(initialPos,stepSize,acceptRate,newPos,accept,targetAcceptRate,stepSizeInc,stepSizeDec,stepSizeMin,stepSizeMax,acceptDecay):
    """
    Perform a HMC move
    :param intialPos: the states of field before evolution
    :param stepSize: the size of a single step
    :param acceptRate: the acceptance rate pass through
    :param newPos: the proposed Pos update
    :param accept: a bool variable of if the udpate is accepted
    :param targetAcceptRate: the desired acceptance rate
    :param stepSizeInc: if the acceptance rate failed targetAcceptRate, the stepSize rise by stepSizeInc
    :param stepSizeDec: if the acceptance rate great than targetAcceptRate, the stepSize decrease by stepSizeDec
    :param stepSizeMin: the minium of stepSize
    :param stepSizeMax: the maxium of stepSize
    :param acceptDecay: the decay rate of old acceptRate
    :return newPos: the states of the field after evolution
    :return newV: the velocity after evolution
    """
    Pos = tf.where(accept,newPos,initialPos)
    newStepsize = tf.multiply(stepSize,tf.where(tf.greater(acceptRate,targetAcceptRate),stepSizeInc,stepSizeDec)) #Update stepSize according to acceptRate
    newStepsize = tf.maximum(tf.minimum(newStepsize,stepSizeMax),stepSizeMin)
    newAcceptRate = tf.add(acceptDecay*acceptRate,(1.0-acceptRate)*tf.reduce_mean(tf.to_float(accept)))
    return Pos,newStepsize,newAcceptRate

class HMCSampler:
    """
    TensorFlow implementation for Hamiltonian Monte Carlo
    """
    def __init__(self,energyFn,prior,stepSize=0.1,steps=10,targetAcceptRate=0.65,acceptDecay=0.9,stepSizeMin=0.001,stepSizeMax=1000,stepSizeDec=0.97,stepSizeInc=1.03):
        self.energyFn = energyFn
        self.prior = prior
        self.z = self.energyFn.z
        self.stepSize = tf.Variable(stepSize)
        self.acceptRate = tf.Variable(targetAcceptRate)
        self.sess = tf.InteractiveSession()
        def fn(zsa,x):
            z,s,a = zsa
            accept,newPos,newV = hmcMove(z,energyFn,stepSize,steps)
            z_,s_,a_ = hmcUpdate(z,s,a,newPos,accept,targetAcceptRate,stepSizeInc,stepSizeDec,stepSizeMin,stepSizeMax,acceptDecay)
            #print(z_)
            #z_,s_,a_ = z+1,s+1,a+1
            return z_,s_,a_
        self.steps = tf.placeholder(tf.int32,[])
        elems = tf.zeros([self.steps])
        self.z_,self.stepSize_,self.acceptRate_ = tf.scan(fn,elems,(self.z,self.stepSize,self.acceptRate),back_prop=False)
        self.sess.run(tf.global_variables_initializer())
    def sample(self,steps,batchSize):
        z,stepSize,acceptRate = self.sess.run([self.z_,self.stepSize_,self.acceptRate_],feed_dict={self.steps:steps,self.z:self.prior(batchSize)})
        #z = np.transpose(z,[1,0,2])
        return z

def main():
    '''
    Test script for hmc
    '''
    from model.doubleGaussian import doubleGaussian as energyFn
    def prior(batchSize):
        return np.random.normal(0,1,[batchSize,2])
    t = energyFn("test")
    hmc = HMCSampler(t,prior)
    z=hmc.sample(8000,1)
    #print(z)
    print(t.measure(z,2))
    #print(hmc.sess.run(hmc.elems,feed_dict={hmc.steps:10}))

if __name__ == "__main__":
    main()