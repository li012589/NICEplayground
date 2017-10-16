if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from NICE.niceLayer import NiceLayer,NiceNetwork
from NICEMC.NICEMC import NICEMCSampler
from utils.autoCorrelation import autoCorrelationTimewithErr
from utils.acceptRate import acceptance_rate

from model.phi4 import phi4
from utils.mlp import mlp

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

'''Define the model to evaluate'''
dim = 3
l = 3
n = l**dim
Kappa = [0.18,0.22]
Lamb = [1.145]


'''define sampler to initialize'''
def prior(batchSize):
    return np.random.normal(0,1,[batchSize,n])

saveName = "phi4_3D"+str(n)+"_"+str(1.145)+"_"+str(0.18)

res = []
errors = []
cond = []
autos=[]
arates=[]
'''Define the same NICE-MC sampler as in training'''
m = 2
b = 8
for kappa in Kappa:
    for lamb in Lamb:
        #mod.reload(n,l,dim,kappa,lamb)
        net = NiceNetwork()
        niceStructure = [([[n,400],[400,800],[800,400],[400,n]],'generator/v1',tf.nn.relu,False),([[n,400],[400,800],[800,400],[400,n]],'generator/x1',tf.nn.relu,True),([[n,400],[400,800],[800,400],[400,n]],'generator/v2',tf.nn.relu,False)]
        discriminatorStructure = [[2*n,400],[400,800],[800,400],[400,1]]

        for dims, name ,active, swap in niceStructure:
            net.append(NiceLayer(dims,mlp,active,name,swap))
        dnet = mlp(discriminatorStructure,leaky_relu,"discriminator")
        mod = phi4(n,l,dim,kappa,lamb,saveName)
        sampler = NICEMCSampler(mod,prior,net,dnet,b,m,'./savedNetwork','./tfSummary')

        '''Starting sampling'''
        TimeStep = 800
        BatchSize = 100
        BurnIn = 300
        bins = 2
        ifload = True

        z,v = sampler.sample(TimeStep,BatchSize,ifload,True)
        z_o = z[BurnIn:,:]
        m_abs = np.mean(z_o,2)
        m_abs = np.absolute(m_abs)

        m_abs_p = np.mean(m_abs)
        autoCorrelation,error =  autoCorrelationTimewithErr(m_abs,bins)
        acceptRate = acceptance_rate(z_o)
        print("kappa:",mod.kappa)
        print("lambda:",mod.lamb)
        res.append(m_abs_p)
        errors.append(error)
        cond.append('l:'+str(lamb)+";"+"k"+str(kappa))
        print("measure: <|m|/V>",m_abs_p,"with error:",error)
        print('Acceptance Rate:',(acceptRate),'Autocorrelation Time:',(autoCorrelation))
        autos.append(autoCorrelation)
        arates.append(acceptRate)
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