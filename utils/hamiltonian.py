import tensorflow as tf

def kineticEnergy(v):
    '''
    Return the value of the kineticEnergy
    :param v: velocity variable
    :return: kineticEnergy
    '''
    return 0.5*tf.reduce_sum(tf.multiply(v,v),axis=1)

def hamiltonian(p,v,f):
    '''
    Return the value of the Hamiltonian
    :param p: position variable
    :param v: velocity variable
    :param f: energy function
    :return: hamiltonian
    '''
    return f(p)+kineticEnergy(v)