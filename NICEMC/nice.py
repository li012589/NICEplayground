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

def dense(inputs, num_outputs, activation_fn=tf.identity, normalizer_fn=None, normalizer_params=None):
    return tcl.fully_connected(inputs, num_outputs, activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)


class Layer(object):
    """
    Base method for implementing flow based models.
    `forward` and `backward` methods return two values:
     - the output of the layer
     - the resulting change of log-determinant of the Jacobian.
    """
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError(str(type(self)))

    def backward(self, inputs):
        raise NotImplementedError(str(type(self)))


class NiceLayer(Layer):
    def __init__(self, dims, name='nice', swap=False):
        """
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: structure of the nice network
        :param name: TensorFlow variable name scope for variable reuse.
        :param swap: Update x if True, or update v if False.
        """
        super(NiceLayer, self).__init__()
        self.dims, self.reuse, self.swap = dims, False, swap
        self.name = 'generator/' + name

    def forward(self, inputs):
        x, v = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, reuse=self.reuse)
            x = x + t
        else:
            t = self.add(x, v_dim, reuse=self.reuse)
            v = v + t
        return [x, v]

    def backward(self, inputs):
        x, v, = inputs
        x_dim, v_dim = x.get_shape().as_list()[-1], v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v, x_dim, reuse=True)
            x = x - t
        else:
            t = self.add(x, v_dim, reuse=True)
            v = v - t
        return [x, v]

    def add(self, x, dx, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            for dim in self.dims:
                x = dense(x, dim, activation_fn=tf.nn.relu)
            x = dense(x, dx)
            return x

    def create_variables(self, x_dim, v_dim):
        assert not self.reuse
        x = tf.zeros([1, x_dim])
        v = tf.zeros([1, v_dim])
        _ = self.forward([x, v])
        self.reuse = True


class NiceNetwork(object):
    def __init__(self, x_dim, v_dim):
        self.layers = []
        self.x_dim, self.v_dim = x_dim, v_dim

    def append(self, layer):
        layer.create_variables(self.x_dim, self.v_dim)
        self.layers.append(layer)

    def forward(self, inputs):
        #x = inputs
        for layer in self.layers:
            inputs= layer.forward(inputs)
        return inputs

    def backward(self, inputs):
        #x = inputs
        for layer in reversed(self.layers):
            inputs= layer.backward(inputs)
        return inputs

    def __call__(self, x, is_backward):
        return tf.cond(
            is_backward,
            lambda: self.backward(x),
            lambda: self.forward(x)
        )

class TrainingOperator(object):
    def __init__(self, network):
        self.network = network

    def __call__(self, inputs, steps):
        def fn(zv, x):
            """
            Transition for training, without Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for training p(v).
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random_normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            z_, v_ = self.network.forward([z, v])
            return z_, v_

        elems = tf.zeros([steps])
        return tf.scan(fn, elems, inputs, back_prop=True)


class InferenceOperator(object):
    def __init__(self, network, energy_fn):
        self.network = network
        self.energy_fn = energy_fn
        self.log = expLogger({})

    def __call__(self, inputs, steps):
        def fn(zv, x):
            """
            Transition with Metropolis-Hastings.
            `z` is the input state.
            `v` is created as a dummy variable to allow output of v_, for debugging purposes.
            :param x: variable only for specifying the number of steps
            :return: next state `z_`, and the corresponding auxiliary variable `v_`.
            """
            z, v = zv
            v = tf.random_normal(shape=tf.stack([tf.shape(z)[0], self.network.v_dim]))
            z_, v_ = self.network([z, v], is_backward=(tf.random_uniform([]) < 0.5))
            ep = hamiltonian(z, v, self.energy_fn)
            en = hamiltonian(z_, v_, self.energy_fn)
            accept = metropolis_hastings_accept(ep, en,self.log)
            z_ = tf.where(accept, z_, z)
            return z_, v_

        elems = tf.zeros([steps])
        return tf.scan(fn, elems, inputs, back_prop=False)

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
