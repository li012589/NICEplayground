import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from model.phi4 import phi4
from hmc.hmc import HMCSampler
from Metropolois.Metropolis import MHSampler
from utils.autoCorrelation import autoCorrelationTime
from utils.acceptRate import acceptance_rate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''Setting model's size'''
zSize = 9

'''define sampler to initialize'''
def prior(bs):
    return np.random.normal(0,1,[bs,zSize])

'''Define the model to evaluate'''
n = 9
dim = 2
l = 3
kappa = 1
lamb = 1
energyFn = phi4(n,l,dim,kappa,lamb)

'''Define sampler'''
hmc = HMCSampler(energyFn,prior)
mh = MHSampler(energyFn,prior)

'''Start sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 2

print("HMC")
z = hmc.sample(TimeStep,BatchSize)
z_o = z[BurnIn:,:]
z_ = np.reshape(z_o,[-1,zSize])
print("mean: ",np.mean(z_))
print("std: ",np.std(z_))
zt = np.mean(z_o,2)
autoCorrelation =  autoCorrelationTime(zt,bins)
acceptRate = acceptance_rate(z_o)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

print("MH")
z_ = mh.sample(TimeStep,BatchSize)
z_o = z_[BurnIn:,:]
z_ = np.reshape(z_o,[-1,zSize])
print("mean: ",np.mean(z_))
print("std: ",np.std(z_))
zt = np.mean(z_o,2)
autoCorrelation =  autoCorrelationTime(zt,bins)
acceptRate = acceptance_rate(z_o)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))