import numpy as np

class model:
    def __init__(self):
        pass
    def __call__(self,z):
        raise NotImplementedError(str(type(self)))
    def mean(self,z,s):
        return np.mean(z[:,s:],axis=1)
    def std(self,z,s):
        return np.std(z[:,s:],axis=1)
    def measure(self,z,n,s):
        return np.mean(np.power(z[:,s:],n),axis=1)