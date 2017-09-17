import tensorflow as tf

def weightVariable(name,shape,init = None):#tf.truncated_normal_initializer(stddev=0.02)):
    initial = tf.get_variable(name,shape,dtype=tf.float32, initializer=init)
    return initial

def biasVariable(name,shape,init = None):#tf.truncated_normal_initializer(stddev=0.02)):
    initial = tf.get_variable(name,shape,initializer=init)
    return initial
