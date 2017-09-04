import numpy as np
import tensorflow as tf

class Ising:
    '''
    A class represnt Ising model
    '''
    def __init__(self,n,l,d,J,mu,name="Ising"):
        self.name = name
        self.d = d
        self.n = n
        self.l = l
        self.J = J
        self.mu = mu
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
                I = tf.zeros_like(tf.slice(z,[0,0],[-1,1]),dtype=tf.float32) #TODO: init dynamically constant tensor
                n = tf.constant(0)
                cc = lambda I,i: i<2*self.d
                def ffn(I,n):
                    #I += tf.cast(tf.slice(z,[0,self.hoppingTable[i][n]],[-1,1])*z[i],tf.float32)
                    n += 1
                    return [I,n]
                I,n = tf.while_loop(cc,ffn,[I,n])
                S += I
                i += 1
                return [S,i]
            S,i = tf.while_loop(c,fn,[S,i])
            S *= self.J
            S = tf.reshape(S,[-1]) - tf.reshape(self.mu*tf.cast(tf.reduce_sum(z,1),tf.float32),[-1])
            return S
    def mean(self,z,s):
        pass
    def std(self,z,s):
        pass
    def measure(self,z,n,s):
        pass

if __name__ == "__main__":
    '''
    Test script
    '''
    def prior(bs,n):
        return np.random.normal(0,1,[bs,n])
    t = Ising(4,2,2,1,1)
    z_ = np.array([[1,2,3,4],[2,3,4,5]])
    print(z_)
    sess = tf.InteractiveSession()
    print(sess.run(t.hoppingTable))
    print(sess.run(t(z_)))