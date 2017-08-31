import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from model.ring2d import Ring2d
from model.doubleGaussian import doubleGaussian
from model.phi4 import phi4
from hmc.hmc import HMCSampler

def prior(bs):
    return np.random.normal(0,1,[bs,9])

if __name__ == "__main__":
    #energyFn = doubleGaussian()
    energyFn = phi4(9,3,2,1,1)
    myhmc = HMCSampler(energyFn,prior)
    z_ = myhmc.sample(80,8)
    z_ = z_[:,30:]
    z_ = np.reshape(z_,[-1,2])
    x_,y_ = z_[:,0],z_[:,1]
    print(np.mean(x_))
    print(np.std(x_))
    #print(z)
