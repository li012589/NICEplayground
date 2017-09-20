import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from model.ring2d import Ring2d
from model.doubleGaussian import doubleGaussian
from model.Ising import Ising
from model.phi4 import phi4
from hmc.hmc import HMCSampler
from Metropolois.Metropolis import MHSampler
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def prior(bs):
    return np.random.normal(0,1,[bs,9])

if __name__ == "__main__":
    #energyFn = Ring2d()
    energyFn = Ising(9,3,2,1,1)
    #energyFn = doubleGaussian()
    #energyFn = phi4(9,3,2,1,1)
    hmc = HMCSampler(energyFn,prior)
    mh = MHSampler(energyFn,prior)
    print("HMC")
    z_ = hmc.sample(8,8)
    print(z_)
    z_ = z_[:,3:]
    z_ = np.reshape(z_,[-1,2])
    x_,y_ = z_[:,0],z_[:,1]
    print(np.mean(x_))
    print(np.std(x_))
    print("MH")
    z_ = mh.sample(8,8)
    z_ = z_[:,3:]
    z_ = np.reshape(z_,[-1,2])
    x_,y_ = z_[:,0],z_[:,1]
    print(np.mean(x_))
    print(np.std(x_))
