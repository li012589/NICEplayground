if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from NICEMC.NICEMC import NICEMCSampler
#from utils.autoCorrelation import autoCorrelationTime
#from utils.acceptRate import acceptance_rate

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

#from utils.parameterInit import weightVariable, biasVariable
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
sampler.train(500,100000,5000,32,1000,100,32,5000,1000,5,32,1000,True,False)