import numpy as np

def gaussian(mu, sig):
    np.exp(-.5*((mu/sig))**2)/(sig*np.sqrt(2*np.pi))

def meanshift(data):
    X = np.copy(data)
    for it in range(5):
        for i,x in enumerate(X):
            dist = np.sqrt(((x-X)**2).sum(1))
            weight = gaussian(dist, 2.5)
            X[i] = (np.expand_dims(weight, 1)*X).sums(0) / weight.sum()
        return X