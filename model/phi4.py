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
        with tf.variable_scope(self.name):
            i = tf.constant(0)
            S = tf.zeros_like(tf.slice(z,[0,0],[-1,1]),dtype=tf.float32) #TODO: init dynamically constant tensor
            c = lambda S,i: i<self.n
            def fn(S,i):
                phin = tf.zeros_like(tf.slice(z,[0,0],[-1,1]),dtype=tf.float32) #TODO: init dynamically constant tensor
                n = tf.constant(0)
                cc = lambda phin,i: i<2*self.d
                def ffn(tmpphin,tmpn):
                    tmpphin += tf.cast(tf.slice(z,[0,self.hoppingTable[i][tmpn]],[-1,1]),dtype=tf.float32)
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
    t = phi4(9,3,2,1,1)
    #z = prior(2,4)
    z = np.array([[1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,10]])
    print(z)
    sess = tf.InteractiveSession()
    #print(sess.run(t.z,feed_dict={t.z:z}))
    print(sess.run(t.hoppingTable))
    i = tf.constant(0)
    j = tf.constant(0)
    #print(sess.run(i))
    tmp = tf.placeholder(tf.float32,[None,4])
    #print(sess.run(tf.slice(tmp,[0,t.hoppingTable[i][j]],[-1,1]),feed_dict={tmp:z}))
    print(sess.run(t(z)))
