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

def metropolisHastingsAccept(energyPre,eneryNext,expLogger,ifuseLogger = False):
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
    pass

def leapfrog(pos,vel,step,energyFn,i):
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
    force = tf.gradients(tf.reduce_sum(energyFn),pos)[0]

def main():
    pass

if __name__ == "__main__":
    main()