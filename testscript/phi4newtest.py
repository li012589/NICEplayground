if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from NICEMC.NICEMC import NICEMCSampler
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptance_rate

from model.phi4 import phi4
from utils.mlp import mlp

'''Define the model to evaluate'''
dim = 3
l = 3
n = l**dim
kappa = 0.18
lamb = 1.145

mod = phi4(n,l,dim,kappa,lamb,"phi4_3D_"+str(n)+"_"+str(lamb)+"_"+str(kappa))

'''Setting model's size'''
zSize = n

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,zSize])

'''Define the same NICE-MC sampler as in training'''
m = 2
b = 8
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
z_o = z[BurnIn:,:]
m_abs = np.mean(z_o,2)
m_abs = np.absolute(m_abs)
m_abs_p = np.mean(m_abs)
autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
acceptRate = acceptance_rate(z_o)
print("kappa:",kappa)
print("measure: <|m|/V>",m_abs_p,"with error:",error)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
