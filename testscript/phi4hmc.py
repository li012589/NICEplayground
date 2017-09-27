if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import numpy as np
from hmc.hmc import HMCSampler
from Metropolois.Metropolis import MHSampler
from utils.autoCorrelation import autoCorrelationTime
from utils.acceptRate import acceptance_rate

from model.phi4 import phi4

'''Setting model's size'''
zSize = 27

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,zSize])

'''Define the model to evaluate'''
n = 27
dim = 3
l = 3
kappa = 1.145
lamb = 0.14
Lamb = [i/100 for i in range(14,22)]
print(Lamb)

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
