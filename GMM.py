import readData
import numpy as np
from sklearn.mixture import GMM

from sklearn.decomposition import PCA
from time import time
import sklearn.metrics as metrics
import accuracy

def estimate_GMM(data, clusters, metric):
    t0 = time()
    model = GMM(n_components=clusters,n_iter = 5,n_init =5)
    model.fit(data)
    labels = model.predict(data)
    t1 = time()
    return ('EM',clusters,accuracy.getAccuracy(data,labels,len(data),metric),t1-t0)