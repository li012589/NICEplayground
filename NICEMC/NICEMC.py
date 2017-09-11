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
#from utils.buff import Buffer

class Buffer:
    def __init__(self,data):
        self.data = data
        #self.perm = np.random.permutation(self.data.shape[0])
        #self.pointer = 0
    def insert(self,data):
        self.data = np.concatenate([self.data,data],axis=0)
        #self.perm = np.random.permutation(self.data.shape[0])
        #self.pointer = 0
    def discard(self,ratio):
        size = self.data.shape[0]
        perm = np.random.permutation(size)
        self.data = self.data[perm[:int(size*(1-ratio))]]
        #self.perm = np.random.permutation(self.data.shap[0])
        #self.pointer = 0
    def __call__(self,batchSize):
        if self.pointer+batchSize >= self.data.shape[0]:
            batchSize = self.data.shape[0]
        perm = np.random.permutation(self.data.shape[0])
        return self.data(perm[:batchSize])

class NiceNetworkOperator:
    def __init__(self,network,energyFn):
        self.network = lambda inputs,t:tf.cond(t>0.5,lambda: network.forward(inputs),lambda: network.backward(inputs))
        self.energyFn = energyFn
        self.explog = expLogger({})
    def __call__(self,inputs,steps,vDim,ifMH):
        if ifMH:
            flag = False
            def fn(zv,step):
                z,v = zv
                #print(z)
                v = tf.random_normal([tf.shape(z)[0],vDim])
                z_,v_ = self.network([z,v],tf.random_uniform([]))
                accept = metropolisHastingsAccept(hamiltonian(z,v,self.energyFn),hamiltonian(z_,v_,self.energyFn),self.explog)
                #accept = tf.convert_to_tensor(np.array([[1,0,1]]),tf.bool)
                #accept = tf.ones_like(z_,tf.bool)
                #accept = tf.zeros_like(z_,tf.bool)
                z_ = tf.where(accept,z_,z)
                return z_,v_
        else:
            flag = True
            def fn(zv,step):
               z, v = zv
               v = tf.random_normal(shape=tf.stack([tf.shape(z)[0],vDim]))
               z_, v_ = self.network([z,v],tf.constant(1,dtype=tf.float32))
               return z_, v_
        elems = tf.zeros([steps])
        return tf.scan(fn,elems,inputs,back_prop=flag)

class NICEMCSampler:
    def __init__(self,energyFn,prior,network,discriminator,b,m,scale=10.0,eta=1.0):
        self.energyFn = energyFn
        self.prior = prior
        self.Operator = NiceNetworkOperator(network,energyFn)
        self.network = network
        self.discriminator = discriminator
        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1), [])) + 1 # Random chose from [1,b]
        self.m = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1
        self.zDim = energyFn.z.get_shape().as_list()[1]
        self.vDim = self.zDim
        self.sess = tf.InteractiveSession()

        self.z = tf.placeholder(tf.float32,[None,self.zDim])
        self.reallyData = tf.placeholder(tf.float32,[None,self.zDim])
        self.batchDate = tf.placeholder(tf.float32,[None,self.zDim])
        #self.buff = Buffer({})

        zBatchSize = tf.shape(self.z)[0]
        rdBatchSize = tf.shape(self.reallyData)[0]

        v = tf.random_normal([zBatchSize,self.zDim])
        self.steps = tf.placeholder(tf.int32,[])
        self.z_,self.v_ = self.Operator((self.z,v),self.steps,self.vDim,True)

        v_ = tf.random_normal([zBatchSize,self.zDim])
        z1,v1 = self.Operator((self.z,v_),self.b,self.vDim,False)
        z1 = z1[-1]
        v1 = v1[-1]
        z1_ = tf.stop_gradient(z1)
        v1_ = tf.random_normal([rdBatchSize,self.zDim])
        z2,v2 = self.Operator((self.reallyData,v1_),self.m,self.vDim,False)
        z2 = z2[-1]
        v2 = v2[-1]
        v2_ = tf.random_normal([zBatchSize,self.zDim])
        z3,v3 = self.Operator((z1_,v2_),self.m,self.vDim,False)
        z3 = z3[-1]
        v3 = v3[-1]

        zConcated = tf.concat([tf.concat([z2,self.reallyData],1),tf.concat([z3,z1],1)],0)
        vConcated = tf.reshape(tf.concat([v1,v2,v3],1),[-1,self.vDim])
        #vConcated = tf.concat([v1,v2,v3],1)
        reallyData = tf.reshape(self.reallyData,[-1,2*self.zDim])
        batchDate = tf.reshape(self.batchDate,[-1,2*self.zDim])

        rD = self.discriminator(reallyData)
        fD = self.discriminator(zConcated)

        epsilon = tf.random_uniform([],0.0,1.0)
        hat = batchDate*epsilon + zConcated*(1-epsilon)
        hatD = self.discriminator(hat)
        gradientHatD = tf.gradients(hatD,hat)[0]
        gradientHatD = tf.norm(gradientHatD)
        gradientHatD = tf.reduce_mean(tf.square(gradientHatD-1.0)*scale)
        self.Dloss = tf.reduce_mean(rD)-tf.reduce_mean(fD)+gradientHatD
        self.Gloss = tf.reduce_mean(fD)+tf.reduce_mean(0.5*tf.multiply(vConcated,vConcated))*eta

        GVar = [var for var in tf.global_variables() if 'generator' in var.name]
        DVar = [var for var in tf.global_variables() if 'discriminator' in var.name]

        #print(self.Gloss)
        #print("GVar")
        #for i in GVar:
            #print(i)

        self.trainD = tf.train.AdamOptimizer().minimize(self.Dloss,var_list=DVar)
        self.trainG = tf.train.AdamOptimizer().minimize(self.Gloss,var_list=GVar)

        self.sess.run(tf.global_variables_initializer())

    def sample(self,steps,batchSize):
        def feed_dict(batchSize):
            return{self.z:self.prior(batchSize),self.reallyData:self.buff(batchSize),self.batchDate:self.buff(batchSize)}
        z,v = self.sess.run([self.z_,self.v_], feed_dict={self.z:self.prior(batchSize),self.steps:steps})
        return z,v
    def train(self,trainSteps,epochSteps,totalEpoch,bootstrapBatchSize,bootstrapBurnIn,logSteps):
        pass
        #for epoch in xrange(totalEpoch):
            #self.sess.run(self.Gloss,feed_dict={feed_dict()})

if __name__ == "__main__":
    '''
    Test script
    '''
    from utils.parameterInit import weightVariable, biasVariable
    from model.TGaussian import TGaussian
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

    def prior(batchSize):
        return np.random.normal(0,1,[batchSize,3])

    mod = TGaussian("test")
    net = NiceNetwork()
    args1 = [([[3,10],[10,5],[5,3]],'generator/v1',False),([[3,10],[10,3]],'generator/x1',True),([[3,4],[4,3]],'generator/v2',False)]
    #args = [([2],'x1',True),([2],'v1',False),([2],'x2',True)]
    for dims, name ,swap in args1:
        net.append(NiceLayer(dims,mlp,tf.nn.relu,name,swap))
    #Operator = NiceNetworkOperator(net,mod)
    #z = np.array([[0,0.1,-0.1],[0,0.1,0.1]])
    #z_ = tf.convert_to_tensor(z,dtype=tf.float32)
    #vDim = 3
    #v_ = tf.random_normal([z_.get_shape().as_list()[0],vDim])
    #inputs = (z_,v_)
    #print("input")
    #print(inputs)
    #steps = tf.constant(10)
    #ret = Operator(inputs,steps,vDim,True)
    #z1_,v1_ = ret
    #ret = tf.reshape(ret,[-1,ret.get_shape().as_list()[-1]])
    #v1_ = tf.random_normal([z_.get_shape().as_list()[0],vDim])
    #print("ret")
    #print(ret)
    sess = tf.InteractiveSession()
    b = 5
    m = 10
    dnet = mlp([[6,40],[40,30],[30,5],[5,1]],tf.nn.relu,"discriminator")
    sampler = NICEMCSampler(mod,prior,net,dnet,b,m)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./test/",graph=tf.get_default_graph())
    summary_writer.flush()

    z,v = sampler.sample(10,1)
    print(z)

    #ret2 = Operator(ret,steps,vDim,True)
    #print(sess.run(ret))
