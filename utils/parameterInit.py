import tensorflow as tf

def weightVariable(name,shape,init = None):
    initial = tf.get_variable(name,shape,dtype=tf.float32, initializer=init)
    return initial

def biasVariable(name,shape,init = None):
    initial = tf.get_variable(name,shape,initializer=init)
    return initial
