if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from NICEMC.NICEMC import NICEMCSampler
from utils.autoCorrelation import autoCorrelationTime
from utils.acceptRate import acceptance_rate

from utils.parameterInit import weightVariable, biasVariable
from model.TGaussian import TGaussian
from model.ring2d import Ring2d
from model.phi4 import phi4
from utils.mlp import mlp

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

s = 2
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,s])

#mod = TGaussian("test")
mod = Ring2d("test")
#mod = phi4(9,3,2,1,1)
net = NiceNetwork()
args1 = [([[s,400],[400,s]],'generator/v1',tf.identity,False),([[s,400],[400,s]],'generator/x1',tf.identity,True),([[s,400],[400,s]],'generator/v2',tf.identity,False)]
#args = [([2],'x1',True),([2],'v1',False),([2],'x2',True)]
for dims, name ,active, swap in args1:
    net.append(NiceLayer(dims,mlp,active,name,swap))
b = 8
m = 2
dnet = mlp([[2*s,400],[400,400],[400,400],[400,1]],leaky_relu,"discriminator")
sampler = NICEMCSampler(mod,prior,net,dnet,b,m,'./savedNetwork','./tfSummary')

z,v = sampler.sample(800,100,True,True)
#print(z)
print(z.shape)
z = z[100:,:]
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

#sampler.train(2,10,10,5,5,2,2,10,5,2,2,2,True,True)
#sampler.train(500,100000,5000,32,1000,100,32,5000,1000,5,32,100,True,False)
