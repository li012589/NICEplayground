import numpy as np
import tensorflow as tf

class phi4:
    '''
    A class represent the phi4 model
    '''
    def __init__(self,n,l,d,kappa,lamb,name="phi4"):
        self.name = name
        self.d = d
        self.n = n
        self.l = l
        self.kappa = kappa
        self.lamb = lamb
        self.hoppingTable = dict()
        self.z = tf.placeholder(tf.float32,[None,n])
        for i in range(n):
            LK = n
            y = i
            self.hoppingTable[i] = {}
            for j in reversed(range(d)):
                LK = int(LK/l)
                xk = int(y/LK)
                y = y-xk*LK
                if xk < l-1:
                    self.hoppingTable[i][j] = i + LK
                else:
                    self.hoppingTable[i][j] = i + LK*(1-l)
                if xk > 0:
                    self.hoppingTable[i][j+d] = i - LK
                else:
                    self.hoppingTable[i][j+d] = i-LK*(1-l)
    def __call__(self,z):
        with tf.variable_scope(self.name):
            pass
            #phi2 = tf.constant(0.0)
            i = tf.constant(0)
            S = tf.zeros_like(tf.slice(z,[0,0],[-1,1]))
            c = lambda S,i: i<self.n
            def fn(S,i):
                #phin = tf.constant(0.0)
                #n = tf.constant(0)
                #cc = lambda phin,i: i<self.d
                #def ffn(tmpphin,tmpn):
                    #tmpphin += tf.slice(self.z,[0,self.hoppingTable[i][tmpn]],[-1,1])
                    #return [tmpphin,tmpn]
                #phin_,n_ = tf.while_loop(cc,ffn,[phin,n])
                x = tf.square(tf.slice(z,[0,i],[-1,1]))
                print(x)
                #S += -2*self.kappa*phin_*tf.slice(self.z,[0,i],[-1,1])+phi2+self.lamb*tf.square(tf.add(phi2,-1.0))
                S += x
                print(S)
                i = tf.add(i,1)
                return [S,i]
            S_,i_ = tf.while_loop(c,fn,[S,i])
            return S_

    def mean(self,z):
        pass
    def std(self,z):
        pass
    def measure(self,z):
        pass
 
if __name__ == "__main__":
    '''
    Test script
    '''
    def prior(bs,n):
        return np.random.normal(0,1,[bs,n])
    t = phi4(4,2,2,1,1)
    print(t.hoppingTable[0][0])
    z = prior(2,4)
    #print(z)
    sess = tf.InteractiveSession()
    #print(sess.run(t.z,feed_dict={t.z:z}))
    #print(sess.run(tf.slice(t.z,[0,1],[-1,1]),feed_dict={t.z:z}))
    tmp = tf.placeholder(tf.float32,[None,4])
    #print(sess.run(t(tmp),feed_dict={tmp:z}))


    shape = tf.shape(tmp)
    print(shape)