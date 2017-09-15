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
from utils.autoCorrelation import autoCorrelationTime
from utils.acceptRate import acceptance_rate
#from utils.buff import Buffer

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Buffer:
    def __init__(self,data):
        self.data = data
    def insert(self,data):
        if self.data.shape[0] == 0:
            self.data = data
        else:
            self.data = np.concatenate([self.data,data],axis=0)
    def discard(self,ratio):
        size = self.data.shape[0]
        perm = np.random.permutation(size)
        self.data = self.data[perm[:int(size*(1-ratio))]]
    def __call__(self,batchSize):
        if batchSize >= self.data.shape[0]:
            batchSize = self.data.shape[0]
        perm = np.random.permutation(self.data.shape[0])
        return self.data[perm[:batchSize]]

class NiceNetworkOperator:
    def __init__(self,network,energyFn):
        self.network = lambda inputs,t:tf.cond(t>=0.5,lambda: network.forward(inputs),lambda: network.backward(inputs))
        self.energyFn = energyFn
        self.explog = expLogger({})
    def __call__(self,inputs,steps,vDim,ifMH):
        if ifMH:
            flag = False
            def fn(zv,step):
                z,v = zv
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
    def __init__(self,energyFn,prior,network,discriminator,b,m,savePath,summaryPath,scale=10.0,eta=1.0):
        self.energyFn = energyFn
        self.prior = prior
        self.Operator = NiceNetworkOperator(network,energyFn)
        self.network = network
        self.discriminator = discriminator
        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1), [])) + 1 # Random chose from [1,b]
        self.m = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1
        self.zDim = energyFn.z.get_shape().as_list()[1]
        self.vDim = self.zDim
        self.savePath = savePath
        self.summaryPath = summaryPath
        self.sess = tf.InteractiveSession()

        self.z = tf.placeholder(tf.float32,[None,self.zDim])
        self.reallyData = tf.placeholder(tf.float32,[None,self.zDim])
        self.batchDate = tf.placeholder(tf.float32,[None,self.zDim])
        self.buff = Buffer(np.array([]))

        zBatchSize = tf.shape(self.z)[0]
        rdBatchSize = tf.shape(self.reallyData)[0]

        v = tf.random_normal([zBatchSize,self.zDim])
        self.steps = tf.placeholder(tf.int32,[])
        self.z_,self.v_ = self.Operator((self.z,v),self.steps,self.vDim,True)
        #self.z_ = tf.transpose(self.z_,[1,0,2])

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
        gradientHatD = tf.norm(gradientHatD,axis=1)
        gradientHatD = tf.reduce_mean(tf.square(gradientHatD-1.0)*scale)
        self.Dloss = tf.reduce_mean(rD)-tf.reduce_mean(fD)+gradientHatD
        self.Gloss = tf.reduce_mean(fD)+tf.reduce_mean(0.5*tf.multiply(vConcated,vConcated))*eta

        GVar = [var for var in tf.global_variables() if 'generator' in var.name]
        DVar = [var for var in tf.global_variables() if 'discriminator' in var.name]

        self.trainD = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5, beta2=0.9).minimize(self.Dloss,var_list=DVar)
        self.trainG = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5, beta2=0.9).minimize(self.Gloss,var_list=GVar)

        self.sess.run(tf.global_variables_initializer())

    def sample(self,steps,batchSize,ifload=False,echo=True):
        if ifload:
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(self.savePath)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(self.sess, checkpoint.model_checkpoint_path)
            elif echo:
                print("Loading failed, starting sampler from random parameter")
        elif echo:
            print("No loading, starting sampler from random parameter")
        z,v = self.sess.run([self.z_,self.v_], feed_dict={self.z:self.prior(batchSize),self.steps:steps})
        return z,v
    def train(self,epochSteps,totalSteps,bootstrapSteps,bootstrapBatchSize,bootstrapBurnIn,logSteps,evlBatchSize,evlSteps,evlBurnIn,dTrainSteps,trainBatchSize,saveSteps,ifSummary = True,ifload = False):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.savePath)
        if ifload:
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(self.sess,checkpoint.model_checkpoint_path)
            else:
                print("Loading failed, staring training from random parameter")
        else:
            print("No loading, starting training from random parameter")
        if ifSummary:
            tfGloss = tf.Variable(0.0)
            tfDloss = tf.Variable(0.0)
            tf.summary.scalar("Gloss", tfGloss)
            tf.summary.scalar("Dloss", tfDloss)
            writer = tf.summary.FileWriter(self.summaryPath, self.sess.graph)
        for t in range(totalSteps):
            if t % epochSteps == 0:
                z,v = self.sample(bootstrapSteps+bootstrapBurnIn,bootstrapBatchSize,False,False)
                z = np.reshape(z[bootstrapBurnIn:,:],[-1,z.shape[-1]])
                self.buff.discard(0.5)
                self.buff.insert(z)
                z,v = self.sample(evlSteps+evlBurnIn,evlBatchSize,False,False)
                z = z[evlBurnIn:,:]
                autoCorrelation = autoCorrelationTime(z,7)
                acceptRate = acceptance_rate(np.transpose(z,[1,0,2]))
                print('At step: ',t,'Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
            if t % logSteps == 0:
                Dloss = self.sess.run(self.Dloss,feed_dict={self.z:self.prior(evlBatchSize),self.reallyData:self.buff(evlBatchSize),self.batchDate:self.buff(4*evlBatchSize)})
                Gloss = self.sess.run(self.Gloss,feed_dict={self.z:self.prior(evlBatchSize),self.reallyData:self.buff(evlBatchSize),self.batchDate:self.buff(4*evlBatchSize)})
                print('At step: ',t,"Discriminator Loss:",(Dloss),"Generator Loss:",(Gloss))
            for i in range(dTrainSteps):
                self.sess.run(self.trainD,feed_dict={self.z:self.prior(trainBatchSize),self.reallyData:self.buff(trainBatchSize),self.batchDate:self.buff(4*trainBatchSize)})
            self.sess.run(self.trainG,feed_dict={self.z:self.prior(trainBatchSize),self.reallyData:self.buff(trainBatchSize),self.batchDate:self.buff(4*trainBatchSize)})
            if t % saveSteps == 0:
                if ifSummary:
                    summary = self.sess.run(tf.summary.merge_all(),feed_dict={tfDloss:Dloss,tfGloss:Gloss})
                    writer.add_summary(summary,t)
                    writer.flush()
                saver.save(self.sess, self.savePath+'/nice', global_step = t)
                print("Net parameter saved")

if __name__ == "__main__":
    '''
    Test script
    '''
    from utils.parameterInit import weightVariable, biasVariable
    from model.TGaussian import TGaussian
    from model.ring2d import Ring2d
    from model.phi4 import phi4
    from utils.mlp import mlp

    s = 2
    def prior(batchSize):
        return np.random.normal(0,1,[batchSize,s])

    #mod = TGaussian("test")
    mod = Ring2d("test")
    #mod = phi4(9,3,2,1,1)
    net = NiceNetwork()
    args1 = [([[s,400],[400,s]],'generator/v1',tf.nn.relu,False),([[s,400],[400,s]],'generator/x1',tf.nn.relu,True),([[s,400],[400,s]],'generator/v2',tf.nn.relu,False)]
    #args = [([2],'x1',True),([2],'v1',False),([2],'x2',True)]
    for dims, name ,active, swap in args1:
        net.append(NiceLayer(dims,mlp,active,name,swap))
    b = 8
    m = 2
    dnet = mlp([[2*s,400],[400,400],[400,400],[400,1]],leaky_relu,"discriminator")
    sampler = NICEMCSampler(mod,prior,net,dnet,b,m,'./savedNetwork','./tfSummary')
    '''
    z,v = sampler.sample(80000,1000)
    #print(z)
    print(z.shape)
    z = z[1000:,:]
    #print(z)
    print(z.shape)
    acceptRate = acceptance_rate(np.transpose(z,[1,0,2]))
    print(acceptRate)
    z_ = np.reshape(z,[-1,2])
    z0,z1 = z_[:,0],z_[:,1]
    print(np.mean(z0))
    print(np.std(z0))

    print(np.mean(z1))
    print(np.std(z1))
    '''

    #sampler.train(2,10,10,5,5,2,2,10,5,2,2,2,True,True)
    sampler.train(500,100000,5000,32,1000,100,32,5000,1000,5,32,1000,True,False)
