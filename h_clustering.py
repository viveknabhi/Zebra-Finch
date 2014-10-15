from sklearn.cluster import AgglomerativeClustering
import accuracy
import readData
import numpy as np
from time import time
from sklearn.decomposition import PCA


def Hierarchical_Cluster(data,n_clusters,metric):
    t0 = time()
    model = AgglomerativeClustering(n_clusters)
    labels = model.fit_predict(data)
    t1 = time()
    score = accuracy.getAccuracy(data,labels,len(data),metric)
    return  ('Hierarchical', n_clusters,score,t1-t0)