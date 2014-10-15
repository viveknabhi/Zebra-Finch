from sklearn.cluster import MeanShift,estimate_bandwidth
import accuracy
import numpy as np
from time import time


def mean_shift(data,metric):
    t0 = time()
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=len(data))
    model = MeanShift(cluster_all=True)
    labels = model.fit_predict(data)

    if np.count_nonzero(labels) != 0:
        score = accuracy.getAccuracy(data,labels,len(data),metric)
    else:
        score = 'None'

    t1 = time()

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    return ('Mean Shift',score,n_clusters_)
    
