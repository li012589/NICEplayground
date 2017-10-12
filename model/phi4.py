import numpy as np
import tensorflow as tf


class phi4:
    '''
    A class represent the phi4 model
    '''
    def __init__(self,n,l,d,kappa,lamb,name="phi4"):
        '''
        Initialize computational graph.
        :param n: int, number of points in configuration
        :param l: int, number of points at sides
        :param d: int, dimensions of configuration
        :param kappa: float,
        :param lamb: float,
        '''
        self.name = name
        self.d = d
        self.n = n
        self.l = l
        self.kappa = kappa
        self.lamb = lamb
        self.hoppingTable = []
        self.z = tf.placeholder(tf.float32,[None,n])
        for i in range(n):
            LK = n
            y = i
            self.hoppingTable.append([])
            for j in reversed(range(d)):
                LK = int(LK/l)
                xk = int(y/LK)
                y = y-xk*LK
                if xk < l-1:
                    self.hoppingTable[i].append(i + LK)
                else:
                    self.hoppingTable[i].append(i + LK*(1-l))
                if xk > 0:
                    self.hoppingTable[i].append(i - LK)
                else:
                    self.hoppingTable[i].append(i-LK*(1-l))
        self.hoppingTable = tf.convert_to_tensor(self.hoppingTable,dtype=tf.int32)
    def __call__(self,z):
        '''
        Calcualte energy of configuration z
        :param z: float matrix, configuration
        '''
        with tf.variable_scope(self.name):
            i = tf.constant(0)
            S = tf.zeros_like(tf.slice(z,[0,0],[-1,1]),dtype=tf.float32) #TODO: init dynamically constant tensor
            c = lambda S,i: i<self.n
            def fn(S,i):
                phin = tf.zeros_like(tf.slice(z,[0,0],[-1,1]),dtype=tf.float32) #TODO: init dynamically constant tensor
                n = tf.constant(0)
                cc = lambda phin,i: i<self.d
                def ffn(tmpphin,tmpn):
                    tmpphin += tf.cast(tf.slice(z,[0,self.hoppingTable[i][2*tmpn]],[-1,1]),dtype=tf.float32)
                    tmpn += 1
                    return [tmpphin,tmpn]
                phin_,n_ = tf.while_loop(cc,ffn,[phin,n])
                #phi2 = tf.cast(tf.square(tf.slice(z,[0,i],[-1,1])),tf.float32)
                S += -2*self.kappa*phin_*tf.cast(tf.slice(z,[0,i],[-1,1]),dtype=tf.float32)#+phi2+self.lamb*tf.square(tf.add(phi2,-1.0))
                i += 1
                return [S,i]
            S_,i_ = tf.while_loop(c,fn,[S,i])
            S_ = tf.reshape(S_,[-1]) +tf.reshape(tf.cast(tf.reduce_sum(tf.square(z),1) + self.lamb*tf.reduce_sum(tf.square(tf.square(z)-1),1),tf.float32),[-1])
            return S_
    def reload(self,n,l,d,kappa,lamb,name=None):
        if name is None:
            pass
        else:
            self.name = name
        self.d = d
        self.n = n
        self.l = l
        self.kappa = kappa
        self.lamb = lamb
    def mean(self,z,s):
        return np.mean(z[:,s:],axis=1)
    def std(self,z,s):
        return np.std(z[:,s:],axis=1)
    def measure(self,z,n,s):
        return np.mean(np.power(z[:,s:],n),axis=1)

if __name__ == "__main__":
    '''
    Test script
    '''
    def prior(bs,n):
        return np.random.normal(0,1,[bs,n])
    t = phi4(9,3,2,1,1)
    #z = prior(2,4)
    z = np.array([[1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,10]])
    #z = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]])
    print(z)
    sess = tf.InteractiveSession()
    #print(sess.run(t.z,feed_dict={t.z:z}))
    print(sess.run(t.hoppingTable))
    i = tf.constant(0)
    j = tf.constant(0)
    #print(sess.run(i))
    tmp = tf.placeholder(tf.float32,[None,4])
    #print(sess.run(tf.slice(tmp,[0,t.hoppingTable[i][j]],[-1,1]),feed_dict={tmp:z}))
    print(sess.run(t(t.z),feed_dict={t.z:z}))
