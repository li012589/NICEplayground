import numpy as np

def acceptance_rate(z):
    cnt = z.shape[0] * z.shape[1]
    for i in range(0, z.shape[0]):
        for j in range(1, z.shape[1]):
            if np.min(np.equal(z[i, j - 1], z[i, j])):
                cnt -= 1
    return cnt / float(z.shape[0] * z.shape[1])
