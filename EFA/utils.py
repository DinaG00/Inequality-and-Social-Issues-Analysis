import numpy as np

def replaceNAN(X):
    means = np.nanmean(a=X, axis=0)
    locs = np.where(np.isnan(X))
    print(locs, type(locs))
    X[locs] = means[locs[1]]
    return X
