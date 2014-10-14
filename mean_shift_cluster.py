from sklearn.cluster import MeanShift
import accuracy
import numpy as np

def mean_shift(data,metric):
    model = MeanShift(bin_seeding=True)
    labels = model.fit_predict(data)

    if np.count_nonzero(labels) != 0:
        score = accuracy.getAccuracy(data,labels,len(data),metric)
    else:
        score = 'None'


    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    return(score,n_clusters_)
    
