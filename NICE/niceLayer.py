import tensorflow as tf
import numpy as np

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

    def netlayer(inputs,num_outputs):
        w = np.array([[1,0],[0,1]])
        b = np.array([[1],[1]])
        w_ = tf.convert_to_tensor(w,dtype=tf.float32)
        b_ = tf.convert_to_tensor(b,dtype=tf.float32)
        ret = tf.matmul(inputs,w_)+b_
        return ret

    #lrelu = leaky_relu
    #def network(x,dim):
    xDim = 2
    vDim = 2
    network = dense
    net = NiceNetwork(xDim,vDim)
    #args = [([1],'v1',False),([1],'x1',True),([1],'v2',False)]
    args = [([1],'x1',True)]
    for dims, name ,swap in args:
        net.append(NiceLayer(dims,netlayer,name,swap))
    z = np.array([[2,3],[1,2]])
    v = z+1
    print(z)
    print(v)
    z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    v_ = tf.convert_to_tensor(v,dtype=tf.float32)
    inputs = [z_,v_]
    #print(inputs)
    ret = net.forward(inputs)
    #print(ret)

    #tmp = netlayer(z_,2)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(ret))
    #print(sess.run(tmp))