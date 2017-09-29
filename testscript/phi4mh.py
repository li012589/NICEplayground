if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import numpy as np
#from hmc.hmc import HMCSampler
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
Kappa = [0.15,0.22]
lamb = 1.145
#Lamb = [0]#[i/100 for i in range(15,22)]
#print(Lamb)
'''Start sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 2

res = []
for kappa in Kappa:
    print(lamb)
    energyFn = phi4(n,l,dim,kappa,lamb)
    '''Define sampler'''
    #hmc = HMCSampler(energyFn,prior)
    #z = hmc.sample(TimeStep,BatchSize)
    mh = MHSampler(energyFn,prior)
    print("kappa",mh.energyFn.kappa)
    print("lamb:",mh.energyFn.lamb)
    z = mh.sample(TimeStep,BatchSize)
    z_o = z[BurnIn:,:]
    print(z_o[0,-10:,:])
    m_abs = np.absolute(z_o)
    m_abs = np.mean(m_abs,2)/zSize
    print(m_abs.shape)
    m_abs_p = np.mean(m_abs)
    print(m_abs_p)
    res.append(m_abs_p)
    autoCorrelation =  autoCorrelationTime(m_abs,bins)
    acceptRate = acceptance_rate(z_o)
    print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

print(res)