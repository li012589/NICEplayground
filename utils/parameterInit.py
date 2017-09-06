import tensorflow as tf

def weightVariable(name,shape):
    initial = tf.get_variable(name,shape,dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    return initial

def biasVariable(name,shape):
    initial = tf.get_variable(name,shape,initializer=tf.constant_initializer(0.01))
    return initial
