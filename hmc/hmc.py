import tensorflow as tf
from utils/expLogger import expLogger
import numpy as np

def kineticEnergy(v):
    '''
    Return the value of the kineticEnergy
    :param v: velocity variable
    :return: kineticEnergy
    '''
    return 0.5*tf.reduce_sum(tf.multiply(v,v),axis=1)

def hamilitonian(p,v,f):
    '''
    Return the value of the Hamiltonian
    :param p: position variable
    :param v: velocity variable
    :param f: energy function
    :return: hamiltonian
    '''
    return f(p)+kineticEnergy(v)

def metropolisHastingsAccept(energyPre,eneryNext,expLogger,ifuseLogger = False):# TODO: maybe move this to the utils folder
    """
    Run Metropolis-Hastings algorithm for 1 step
    :param energyPrev: the original energy from before
    :param energyNext: the energy after evolution
    :return: Tensor of boolean values, indicating accept or reject
    """
    energyDiff = energyPre - eneryNext
    if ifuseLogger:
        return expLogger.cal(energyDiff) # TODO: The energyDiff is a tf tensor object, need some workaround
    return (tf.exp(energyDiff)) - tf.random_uniform(tf.shape(energyPre)) >= 0.0

def leapfrogMeta(pos,vel,step,energyFn,i):
    '''
    The leapfrog integrator
    :param pos: the states of the field
    :param vel: the velocity of the field
    :param step: step size of a step
    :param energyFn: the function describe Hamiliton
    :param i: the flag variable contain integration times
    :return newPos: the pos after integration
    :return newV: the velocity after integration
    :return i: the flag variable contain integration times
    '''
    force = tf.gradients(tf.reduce_sum(energyFn(pos)),pos)[0]
    newV = initialV - step*force
    newPos = initialPos +step*newV
    tf.add(i,1)
    return [newPos,newV,i]

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
    newPos,newV,_ = tf.while_loop(tf.less(i,steps),leapfrogMeta,[tmpPos,tmpV,stepSize,i])
    force = tf.gradients(tf.reduce_sum(energyFn(newPos)),newPos)[0]
    newV -= 0.5*stepSize*force
    newPos += stepSize*newV
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
    accept = metropolisHastingsAccept()

def main():
    pass

if __name__ == "__main__":
    main()