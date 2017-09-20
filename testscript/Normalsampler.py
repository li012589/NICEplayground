import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from model.ring2d import Ring2d
from model.TGaussian import TGaussian
#from model.doubleGaussian import doubleGaussian
#from model.Ising import Ising
from model.phi4 import phi4
from hmc.hmc import HMCSampler
from Metropolois.Metropolis import MHSampler
from utils.autoCorrelation import autoCorrelationTime
from utils.acceptRate import acceptance_rate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''Setting model's size'''
zSize = 2

'''define sampler to initialize'''
def prior(bs):
    return np.random.normal(0,1,[bs,zSize])
def priorPhi4(bs):
    return np.random.normal(0,1,[bs,zSize])

'''Define the model to evaluate'''
energyFn = Ring2d()
#energyFn = phi4(9,3,2,1,1)

'''Define sampler'''
hmc = HMCSampler(energyFn,prior)
mh = MHSampler(energyFn,prior)

'''Start sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 7

print("HMC")
z = hmc.sample(TimeStep,BatchSize)
z_o = z[BurnIn:,:]
z_ = np.reshape(z_o,[-1,2])
z1_,z2_ = z_[:,0],z_[:,1]
print("mean: ",np.mean(z1_))
print("std: ",np.std(z1_))
autoCorrelation =  autoCorrelationTime(z_,bins)
acceptRate = acceptance_rate(z_o)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

print("MH")
z_ = mh.sample(TimeStep,BatchSize)
z_o = z_[BurnIn:,:]
z_ = np.reshape(z_o,[-1,2])
z1_,z2_= z_[:,0],z_[:,1]
print("mean: ",np.mean(z1_))
print("std: ",np.std(z1_))
autoCorrelation =  autoCorrelationTime(z_,bins)
acceptRate = acceptance_rate(z_o)
print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
