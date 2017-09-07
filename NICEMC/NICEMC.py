if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from utils.MetropolisHastingsAccept import metropolisHastingsAccept
from utils.hamiltonian import hamiltonian
from utils.expLogger import expLogger

class NiceNetworkOperator:
    def __init__(self,network,energyFn):
        self.network = lambda inputs,t: tf.cond(t>0.5,lambda: network.forward(inputs),lambda: network.backward(inputs))
        self.energyFn = energyFn
        self.explog = expLogger({})
    def __call__(self,inputs,steps,vDim,ifMH):
        if ifMH:
            def fn(zv,step):
                z,v = zv
                v = tf.random_normal([z.get_shape().as_list()[0],vDim])
                z_,v_ = self.network([z,v],tf.random_uniform([]))
                accept = metropolisHastingsAccept(hamiltonian(z,v,self.energyFn),hamiltonian(z_,v_,self.energyFn),self.explog)
                z_ = tf.where(accept,z_,z)
                return z_,v_
        else:
            def fn(zv,step):
               z, v = zv
               v = tf.random_normal(shape=tf.stack([tf.shape(z)[0],vDim]))
               z_, v_ = self.network([z,v],tf.constant(1,dtype=tf.float32))
               return z_, v_
        elems = tf.zeros([steps])
        return tf.scan(fn,elems,inputs,back_prop=False)

class NICEMCSampler:
    def __init__(self,energyFn,prior,network,discriminator,b,m):
        self.energyFn = energyFn
        self.prior = prior
        self.Operator = NiceNetworkOperator(network,energyFn)
        self.network = network
        self.discriminator = discriminator
        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1), [])) + 1 # Random chose from [1,b]
        self.m = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1
        self.zDim = energyFn.z.get_shape().as_list()[1]
        self.vDim = self.xDim
        self.sess = tf.InteractiveSession()

        self.z = tf.placeholder(tf.float32,[None,self.zDim])
        self.reallyData = tf.placeholder(tf.float32,[None,self.zDim])
        self.batchDate = tf.placeholder(tf.float32,[None,self.zDim])

        zBatchSize = tf.shape(self.z)[0]
        rdBatchSize = tf.shape(self.reallyData)[0]

        v = tf.random_normal(zBatchSize,self.zDim)
        self.steps = tf.placeholder(tf.int32,[])
        self.z_,self.v_ = self.Operator((self.z,v),self.steps,True)

        v_ = tf.random_normal([zBatchSize,self.zDim])
        z1,v1 = self.Operator((self.z,v_),self.b)
        z1 = z1[-1]
        v1 = v1[-1]
        z1_ = tf.stop_gradient(z1)
        v1_ = tf.random_normal([rdBatchSize,self.zDim])
        z2,v2 = self.Operator((self.x,v1_),self.m)
        z2 = z2[-1]
        v2 = v2[-1]
        v2_ = tf.random_normal([zBatchSize,self.zDim])
        z3,v3 = self.Operator((z1_,v2_),self.m)
        z3 = z3[-1]
        v3 = v3[-1]

        zConcated = tf.concat([tf.concat([z2,self.x],1),tf.concat([z3,z1],1)],0)
        #vConcated = tf.reshape(tf.concat([v1,v2,v3],1),[-1,self.vDim])
        vConcated = tf.concat([v1,v2,v3],1)

    def sample(self,steps,batchSize):
        pass
    def train(self):
        pass

if __name__ == "__main__":
    '''
    Test script
    '''
    from utils.parameterInit import weightVariable, biasVariable
    from model.TGaussian import TGaussian

    mod = TGaussian("test")

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
    args1 = [([10,5,3],'v1',False),([10,10],'x1',True),([10,10],'v2',False)]
    #args = [([2],'x1',True),([2],'v1',False),([2],'x2',True)]
    for dims, name ,swap in args1:
        net.append(NiceLayer(dims,randomLayer,name,swap))
    Operator = NiceNetworkOperator(net,mod)
    z = np.array([[1,2,3],[2,3,4]])
    z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    vDim = 3
    v_ = tf.random_normal([z_.get_shape().as_list()[0],vDim])
    inputs = (z_,v_)
    steps = tf.constant(10)
    ret = Operator(inputs,steps,vDim,True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(ret))