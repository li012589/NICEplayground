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

zDim = 2

def prior(bs):
    return np.random.normal(0,1,[bs,zDim])
def priorPhi4(bs):
    return np.random.normal(0,1,[bs,9])

if __name__ == "__main__":
    energyFn = Ring2d()
    #energyFn = phi4(9,3,2,1,1)
    hmc = HMCSampler(energyFn,prior)
    mh = MHSampler(energyFn,prior)
    print("HMC")
    z = hmc.sample(800,100)
    z_o = z[300:,:]
    z_ = np.reshape(z_o,[-1,2])
    z1_,z2_ = z_[:,0],z_[:,1]
    print(np.mean(z1_))
    print(np.std(z1_))
    print("autoCorrelationTime:")
    print(autoCorrelationTime(z_,7))
    #print(autoCorrelationTime(z1_,7))
    print("acceptance:")
    #print(z_o)
    print(acceptance_rate(z_o))#np.transpose(z_o,[1,0,2])))
    #print(autoCorrelationTime(z2_,7))
    #print(autoCorrelationTime(z3_,7))
    print("MH")
    z_ = mh.sample(800,100)
    z_o = z_[300:,:]
    z_ = np.reshape(z_o,[-1,2])
    z1_,z2_= z_[:,0],z_[:,1]
    print(np.mean(z1_))
    print(np.std(z1_))
    print("autoCorrelationTime:")
    print(autoCorrelationTime(z_,7))
    #print(autoCorrelationTime(z1_,7))
    print("acceptance:")
    print(acceptance_rate(z_o))#np.transpose(z_o,[1,0,2])))
    #print(autoCorrelationTime(z2_,7))
    #print(autoCorrelationTime(z3_,7))