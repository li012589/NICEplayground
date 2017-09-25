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

from model.TGaussian import TGaussian
from model.ring2d import Ring2d
from model.phi4 import phi4
from utils.mlp import mlp

'''Setting model's size'''
zSize = 2

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,zSize])

'''Define the same NICE-MC sampler as in training'''
#mod = TGaussian()
mod = Ring2d()
#mod = phi4(9,3,2,1,1)

m = 2
b = 8
ifload = False
ifsummary = True
net = NiceNetwork()
niceStructure = [([[zSize,400],[400,zSize]],'generator/v1',tf.nn.relu,False),([[zSize,400],[400,zSize]],'generator/x1',tf.nn.relu,True),([[zSize,400],[400,zSize]],'generator/v2',tf.nn.relu,False)]
discriminatorStructure = [[2*zSize,400],[400,400],[400,400],[400,1]]

for dims, name ,active, swap in niceStructure:
    net.append(NiceLayer(dims,mlp,active,name,swap))
dnet = mlp(discriminatorStructure,leaky_relu,"discriminator")
sampler = NICEMCSampler(mod,prior,net,dnet,b,m,'./savedNetwork','./tfSummary')

'''Starting sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 2
ifload = True

z,v = sampler.sample(TimeStep,BatchSize,ifload,True)
z = z[BurnIn:,:]
z_ = np.reshape(z,[-1,zSize])
z0,z1 = z_[:,0],z_[:,1]
print("mean:",(np.mean(z0)))
print("std:",np.std(z0))
#print(np.mean(z1))
#print(np.std(z1))
autocorrelation = autoCorrelationTime(z[:,:,0],bins)
acceptRate = acceptance_rate(z)
print("autoCorrelation: ",autocorrelation,"acceptRate: ",acceptRate)
