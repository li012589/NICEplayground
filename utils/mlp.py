if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())

import tensorflow as tf
from utils.parameterInit import weightVariable, biasVariable

class mlp:
    def __init__(self,dims,active,name="MLP"):
        self.W = []
        self.B = []
        self.active = active
        with tf.variable_scope(name):
            for i,dim in enumerate(dims):
                self.W.append(weightVariable(str(i)+"wFC",dim))
                self.B.append(biasVariable(str(i)+"bFC",dim[-1]))

    def __call__(self,fc):
        for i,w in enumerate(self.W):
            if i != len(self.W)-1:
                fc = self.active(tf.matmul(fc,w))+self.B[i]
            else:
                fc = (tf.matmul(fc,w)+self.B[i])
        return fc

if __name__ == "__main__":
    import numpy as np
    net = mlp([[2,40],[40,2]],tf.nn.relu,"test")
    print(net.W)
    i = np.array([[-2,-1],[-1,-3],[-2,-1],[-5,-1]])
    i_ = tf.convert_to_tensor(i,dtype=tf.float32)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    ret = sess.run(net(i_))
    print(ret)
    print(sess.run(net.W))
    print(sess.run(net.B))