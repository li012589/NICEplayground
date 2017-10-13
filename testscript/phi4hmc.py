if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from hmc.hmc import HMCSampler
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptance_rate

from model.phi4 import phi4


'''Define the model to evaluate'''
dim = 3
l = 6
n = l**dim
Lamb = [1.145]
Kappa = [i/100 for i in range(15,22+1)]

'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,n])
#print(Lamb)
'''Start sampling'''
TimeStep = 800
BatchSize = 100
BurnIn = 300
bins = 2

res = []
errors = []
cond = []
autos=[]
arates=[]
energyFn = phi4(n,l,dim,0.15,1.145)

for lamb in Lamb:
    for kappa in Kappa:
        '''Define sampler'''
        energyFn.reload(n,l,dim,kappa,lamb)
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
        errors.append(error)
        cond.append('l:'+str(lamb)+";"+"k"+str(kappa))
        autos.append(autoCorrelation)
        arates.append(acceptRate)
        print("kappa:",kappa)
        print("measure: <|m|/V>",m_abs_p,"with error:",error)
        errors.append(error)
        print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
        tf.reset_default_graph()

print("Condition:")
print(cond)
print("measure: <|m|/V>")
print(res)
print("Autorrelation Time:")
print(autos)
print("Acceptance Rate:")
print(arates)
print("Errors:")
print(errors)