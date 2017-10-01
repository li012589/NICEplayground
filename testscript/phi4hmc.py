if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import numpy as np
from hmc.hmc import HMCSampler
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptance_rate

from model.phi4 import phi4

'''Setting model's size'''
zSize = 216

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,zSize])

'''Define the model to evaluate'''
n = 216
dim = 3
l = 6
lamb = 1.145
KAPPA = [i/100 for i in range(15,22+1)]
#print(Lamb)
'''Start sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 2

res = []
errors = []
for kappa in KAPPA:
    energyFn = phi4(n,l,dim,kappa,lamb)
    '''Define sampler'''
    hmc = HMCSampler(energyFn,prior)
    z = hmc.sample(TimeStep,BatchSize)
    z_o = z[BurnIn:,:]
    #print(z_o[0,-10:,:])
    m_abs = np.mean(z_o,2)
    m_abs = np.absolute(m_abs)
    #print(m_abs.shape)
    m_abs_p = np.mean(m_abs)
    res.append(m_abs_p)
    autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
    acceptRate = acceptance_rate(z_o)
    print("kappa:",kappa)
    print("measure: <|m|/V>",m_abs_p,"with error:",error)
    errors.append(error)
    print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))

print("measure: <|m|/V>")
print(res)
print("error:")
print(errors)