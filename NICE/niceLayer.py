import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

class NiceLayer:
    def __init__(self,dims,network,name="niceLayer",swap=False):
        '''
        NICE Layer that takes in [x, v] as input and updates one of them.
        Note that for NICE, the Jacobian is always 1; but we keep it for
        possible extensions to real NVP based flow models.
        :param dims: int matrix, structure of the inner nice network
        :param name: string, TensorFlow variable name scope for variable reuse.
        :param swap: bool, Update x if True, or update v if False.
        '''
        self.dims = dims
        self.name = name
        self.swap = swap
        self.network = network
    def forward(self,inputs):
        x,v = inputs
        xDim = x.get_shape().as_list()[-1]
        vDim = v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v,xDim)
            x = x + t
        else:
            t = self.add(x,vDim)
            v = v + t
        return [x,v]
    def backward(self,inputs):
        x,v = inputs
        xDim = x.get_shape().as_list()[-1]
        vDim = v.get_shape().as_list()[-1]
        if self.swap:
            t = self.add(v,xDim)
            x = x - t
        else:
            t = self.add(x,vDim)
            v = v - t
        return [x,v]
    def add(self,x,xDim):
        with tf.variable_scope(self.name):
            for dim in self.dims:
                x = self.network(x,dim)
            x = self.network(x,xDim)
            return x


class NiceNetwork:
    def __init__(self,xDim,vDim):
        self.layers = []
        self.xDim = xDim
        self.vDim = vDim
    def append(self,layer):
        self.layers.append(layer)
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backward(self,inputs):
        for layer in self.layers:
            inputs = layer.backward()
        return inputs

if __name__ == "__main__":
    import tensorflow.contrib.layers as tcl
    def leaky_relu(x, alpha=0.2):
        return tf.maximum(tf.minimum(0.0, alpha * x), x)

    def dense(inputs, num_outputs, activation_fn=leaky_relu, normalizer_fn=None, normalizer_params=None):
        return tcl.fully_connected(inputs, num_outputs, activation_fn=activation_fn,normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

    #lrelu = leaky_relu
    #def network(x,dim):
    xDim = 2
    vDim = 2
    network = dense
    net = NiceNetwork(xDim,vDim)
    args = [([400],'v1',False),([400],'x1',True),([400],'v2',False)]
    for dims, name ,swap in args:
        net.append(NiceLayer(dims,dense,name,swap))
    z = np.array([[2,3],[1,2]])
    v = z+1
    print(z)
    print(v)
    z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    v_ = tf.convert_to_tensor(v,dtype=tf.float32)
    inputs = [z_,v_]
    print(inputs)
    ret = net.forward(inputs)
    print(ret)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(ret))