import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from model.ring2d import Ring2d
from model.TGaussian import TGaussian
#from model.doubleGaussian import doubleGaussian
#from model.Ising import Ising
#from model.phi4 import phi4
from hmc.hmc import HMCSampler
from Metropolois.Metropolis import MHSampler
from utils.autoCorrelation import autoCorrelationTime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def prior(bs):
    return np.random.normal(0,1,[bs,3])

if __name__ == "__main__":
    energyFn = TGaussian()
    hmc = HMCSampler(energyFn,prior)
    mh = MHSampler(energyFn,prior)
    print("HMC")
    z = hmc.sample(8000,800)
    z_ = z[:,3000:]
    z_ = np.reshape(z_,[-1,3])
    z1_,z2_,z3_ = z_[:,0],z_[:,1],z_[:,2]
    print(np.mean(z1_))
    print(np.std(z1_))
    print("autoCorrelationTime:")
    print(autoCorrelationTime(z1_,7))
    print(autoCorrelationTime(z2_,7))
    print(autoCorrelationTime(z3_,7))
    print("MH")
    z_ = mh.sample(8000,800)
    z_ = z_[:,3000:]
    z_ = np.reshape(z_,[-1,3])
    z1_,z2_,z3_ = z_[:,0],z_[:,1],z_[:,2]
    print(np.mean(z1_))
    print(np.std(z1_))
    print("autoCorrelationTime:")
    print(autoCorrelationTime(z1_,7))
    print(autoCorrelationTime(z2_,7))
    print(autoCorrelationTime(z3_,7))