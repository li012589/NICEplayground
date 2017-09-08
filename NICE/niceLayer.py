if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np

class NiceLayer:
    def __init__(self,dims,network,active,name="niceLayer",swap=False):
        '''
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: int matrix, structure of the inner nice network
        :param name: string, TensorFlow variable name scope for variable reuse.
        :param swap: bool, Update x if True, or update v if False.
        '''
        #self.dims = dims
        self.name = name
        self.swap = swap
        self.network = network(dims,active,name+"/innerNiceLayer")
    def forward(self,inputs):
        x,v = inputs
        if self.swap:
            t = self.network(v)
            x = x + t
        else:
            t = self.network(x)
            v = v + t
        return [x,v]
    def backward(self,inputs):
        x,v = inputs
        if self.swap:
            t = self.network(v)
            x = x - t
        else:
            t = self.network(x)
            v = v - t
        return [x,v]

class NiceNetwork:
    def __init__(self):
        self.layers = []
    def append(self,layer):
        self.layers.append(layer)
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backward(self,inputs):
        for layer in self.layers:
            inputs = layer.backward(inputs)
        return inputs

if __name__ == "__main__":
    '''
    Test script
    '''
    from utils.parameterInit import weightVariable, biasVariable
    from utils.mlp import mlp

    def Fixlayer(inputs, num_outputs,name,reuseMark):
        #print(name)
        w = np.array([[1,0],[0,1]])
        b = np.array([[1],[1]])
        w_ = tf.convert_to_tensor(w,dtype=tf.float32)
        b_ = tf.convert_to_tensor(b,dtype=tf.float32)
        ret = tf.matmul(inputs,w_)+b_
        return ret

    def randomLayer(inputs,num_outputs,name,reuseMark = None):
        with tf.variable_scope(name,reuse=reuseMark):
            wFC = weightVariable("RandomLayerFCw",[inputs.get_shape()[-1],num_outputs])
            bFC = biasVariable("RandomLayerFC",[num_outputs])
            fc = tf.matmul(inputs,wFC)+bFC
            fc = tf.nn.relu(fc)
            return fc

    net = NiceNetwork()
    args1 = [([[2,10],[10,5],[5,2]],'v1',False),([[2,10],[10,2]],'x1',True),([[2,4],[4,2]],'v2',False)]
    #args = [([2],'x1',True),([2],'v1',False),([2],'x2',True)]
    for dims, name ,swap in args1:
        net.append(NiceLayer(dims,mlp,tf.nn.relu,name,swap))
    z = np.array([[2,3],[1,2]],dtype=np.float32)
    v = z+1
    inputs=[z,v]
    print(inputs)
    z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    v_ = tf.convert_to_tensor(v,dtype=tf.float32)
    inputs_ = [z_,v_]
    ret = net.forward(inputs_)
    ret2 = net.forward(inputs_)
    ret2_ = net.backward(ret2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    forward = (sess.run(ret))
    print("forwarding")
    zp_ = tf.convert_to_tensor(forward[0],dtype=tf.float32)
    vp_ = tf.convert_to_tensor(forward[1],dtype=tf.float32)
    forward_ = [zp_,vp_]
    retp = net.backward(forward_)
    backward = sess.run(retp)
    print("backwarding")
    print(backward)
    assert (np.allclose(inputs,backward)), "Fixlayer: input doesn't match backward"
    print("Input matched backward")
    print("2nd test:")
    print("forward")
    print(sess.run(ret2))
    print("backward")
    ret2_r = sess.run(ret2_)
    assert (np.allclose(inputs,ret2_r)), "Fixlayer: input doesn't match backward"
