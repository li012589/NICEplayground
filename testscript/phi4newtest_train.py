if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from NICEMC.NICEMC import NICEMCSampler

from model.phi4 import phi4
from utils.mlp import mlp

'''Define the model to evaluate'''
dim = 3
l = 3
n = l**dim
kappa = 0.18
lamb = 1.145

mod = phi4(n,l,dim,kappa,lamb,"phi4_3D"+str(n)+"_"+str(lamb)+"_"+str(kappa))

'''Setting model's size'''
zSize = n

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,zSize])

'''Define the NICE-MC sampler'''
m = 2
b = 8
ifload = False
ifsummary = True
net = NiceNetwork()
niceStructure = [([[zSize,400],[400,800],[800,400],[400,zSize]],'generator/v1',tf.nn.relu,False),([[zSize,400],[400,800],[800,400],[400,zSize]],'generator/x1',tf.nn.relu,True),([[zSize,400],[400,800],[800,400],[400,zSize]],'generator/v2',tf.nn.relu,False)]
discriminatorStructure = [[2*zSize,400],[400,800],[800,400],[400,1]]

for dims, name ,active, swap in niceStructure:
    net.append(NiceLayer(dims,mlp,active,name,swap))
dnet = mlp(discriminatorStructure,leaky_relu,"discriminator")

'''Configure to use customized session'''
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

sampler = NICEMCSampler(mod,prior,net,dnet,b,m,'./savedNetwork','./tfSummary',sess=sess)

'''Start training'''
print("Training NICE for "+sampler.energyFn.name)
sampler.train(500,100000,5000,32,1000,100,32,5000,1000,5,32,1000,ifsummary,ifload)