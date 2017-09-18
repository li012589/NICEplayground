if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from utils.MetropolisHastingsAccept import metropolisHastingsAccept as metropolis_hastings_accept
from utils.hamiltonian import hamiltonian
#from NICEME.discriminator import dense
from utils.expLogger import expLogger
from utils.parameterInit import weightVariable,biasVariable

def dense(inputs, num_outputs, activation_fn=tf.identity, normalizer_fn=None, normalizer_params=None):
    return tcl.fully_connected(inputs, num_outputs, activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)
class mlpnet:
    def __init__(self,dims,active,name):
        self.name = name+'mlpnet'
        self.dims = [dim[1] for dim in dims][:-1]
        self.outdim = dims[0][0]
        #x = tf.zeros([1,dims[0][0]])
        #_ = self.cal(x,None)
        self.W=[]
        self.B=[]
        self.active = active
        with tf.variable_scope(name):
            for i,dim in enumerate(dims):
                self.W.append(weightVariable(str(i)+"ttwfc",dim))
                self.B.append(biasVariable(str(i)+"ttbfc",dim[1]))
    def cal(self,x,reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            '''
            for dim in self.dims:
                x = dense(x, dim, activation_fn=tf.nn.relu)
            x = dense(x, self.outdim)
            '''
            for i in range(len(self.W)):
                x = tf.matmul(x,self.W[i])+self.B[i]
                if i != len(self.W)-1:
                    x = self.active(x)
            return x
class mlp:
    def __init__(self,dims,active,name="MLP"):
        self.W = []
        self.B = []
        self.active = active
        with tf.variable_scope(name):
            for i,dim in enumerate(dims):
                self.W.append(weightVariable(str(i)+"wFC",dim))
                self.B.append(biasVariable(str(i)+"bFC",dim[-1]))

    def cal(self,fc):
        for i,w in enumerate(self.W):
            if i != len(self.W)-1:
                fc = self.active(tf.matmul(fc,w)+self.B[i])
            else:
                fc = (tf.matmul(fc,w)+self.B[i])
        return fc

class NiceLayer:
    def __init__(self, dims, _,active,name='nice', swap=False):
        """
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: structure of the nice network
        :param name: TensorFlow variable name scope for variable reuse.
        :param swap: Update x if True, or update v if False.
        """
        self.swap = swap
        self.name = name
        self.network = mlp(dims,active,name+"i")
    def forward(self,inputs):
        x, v = inputs
        if self.swap:
            t = self.network.cal(v)
            x = x + t
        else:
            t = self.network.cal(x)
            v = v + t
        return [x, v]
    def backward(self,inputs):
        x, v, = inputs
        if self.swap:
            t = self.network.cal(v)
            x = x - t
        else:
            t = self.network.cal(x)
            v = v - t
        return [x, v]


class NiceNetwork(object):
    def __init__(self, x_dim, v_dim):
        self.layers = []
    def append(self, layer):
        self.layers.append(layer)
    def forward(self, inputs):
        for layer in self.layers:
            inputs= layer.forward(inputs)
        return inputs
    def backward(self, inputs):
        for layer in reversed(self.layers):
            inputs= layer.backward(inputs)
        return inputs

if __name__ == "__main__":
    import numpy as np

    def create_nice_network(x_dim, v_dim, args):
        net = NiceNetwork(x_dim, v_dim)
        for dims, name, swap in args:
            net.append(NiceLayer(dims, name, swap))
        return net

    net = create_nice_network(
            2, 2,
            [
            ([5], 'v1', False),
                ([5], 'x1', True),
                ([5], 'v2', False),
            ]
        )
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
    print(forward)
    zp_ = tf.convert_to_tensor(forward[0],dtype=tf.float32)
    vp_ = tf.convert_to_tensor(forward[1],dtype=tf.float32)
    forward_ = [zp_,vp_]
    retp = net.backward(forward_)
    backward = sess.run(retp)
    print("backwarding")
    print(backward)
    assert (np.allclose(inputs,backward)), "Input doesn't match backward"
    print("Input matched backward")
    print("2nd test:")
    print("forward")
    ret2r = sess.run(ret2)
    print(ret2r)
    print("backward")
    ret2_r = sess.run(ret2_)
    print(ret2_r)
    assert (np.allclose(inputs,ret2_r)), "Input doesn't match backward"
    assert (np.allclose(forward,ret2r)), "First forward doesn't math second forward, the parameter in inner layer may not be same'"
    net2 = create_nice_network(
            1, 1,
            [
            ([5], 'v1x', False),
                ([5], 'x1x', True),
                ([5], 'vx2', False),
            ]
        )
    z = np.array([[-1]],dtype = np.float32)
    v = z+1
    inputs=[z,v]
    print(inputs)
    z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    v_ = tf.convert_to_tensor(v,dtype=tf.float32)
    inputs_ = [z_,v_]
    ret = net2.forward(inputs_)
    ret2 = net2.forward(inputs_)
    ret2_ = net2.backward(ret2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    forward = (sess.run(ret))
    print("forwarding")
    print(forward)
    zp_ = tf.convert_to_tensor(forward[0],dtype=tf.float32)
    vp_ = tf.convert_to_tensor(forward[1],dtype=tf.float32)
    forward_ = [zp_,vp_]
    retp = net2.backward(forward_)
    backward = sess.run(retp)
    print("backwarding")
    print(backward)
    assert (np.allclose(inputs,backward)), "Input doesn't match backward"
    print("Input matched backward")
    print("2nd test:")
    print("forward")
    ret2r = sess.run(ret2)
    print(ret2r)
    print("backward")
    ret2_r = sess.run(ret2_)
    print(ret2_r)
    assert (np.allclose(inputs,ret2_r)), "Input doesn't match backward"
    assert (np.allclose(forward,ret2r)), "First forward doesn't math second forward, the parameter in inner layer may not be same'"

    print("Computing gradients")
    g = sess.run(tf.gradients(ret,inputs_))
    g0 = sess.run(tf.gradients(ret[0],inputs_))
    g1 = sess.run(tf.gradients(ret[1],inputs_))
    g00 = sess.run(tf.gradients(ret[0],inputs_[0]))
    g01 = sess.run(tf.gradients(ret[0],inputs_[1]))
    g10 = sess.run(tf.gradients(ret[1],inputs_[0]))
    g11 = sess.run(tf.gradients(ret[1],inputs_[1]))
    #print(g)
    print(g0)
    print(g1)
    det = g00[0]*g11[0]-g01[0]*g10[0]
    print(det)
    assert (np.isclose(1,det)) , "Determinant of Jacobian is not equal 1"
