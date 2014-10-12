import readData
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from time import time
import sklearn.metrics as metrics
import accuracy


def euclidian_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print 'name', 'time' ,'silhouette_score' 
    print('% 9s   %.2fs    %i   %.3f '
          % (name, (time() - t0), estimator.inertia_,accuracy.getAccuracy(data,estimator.labels_,len(data),'euclidean')))



def doPaddedKmeans():
	data = readData.loadDataFromPickle('mfccDumpPadded.p')
	listData = []
	for item in data:
		listData.append(data[item].flatten())

	listData = np.array(listData)
	print listData.shape
	for i in xrange(4,10,1):
		listData = PCA(n_components=2).fit_transform(listData)
		kmeans = KMeans(init='k-means++', n_clusters=i+1, n_init=1)
		euclidian_k_means(kmeans, 'kmeans++', listData)


def doAvgCoeffKmeans():
	data = readData.loadDataFromPickle('mfccDump.p')
	listData = readData.getAveragedCepstralCoefficients(data)
	print listData.shape
	for i in xrange(4,10,1):
		listData = PCA(n_components=2).fit_transform(listData)
		kmeans = KMeans(init='k-means++', n_clusters=i+1, n_init=1)
		euclidian_k_means(kmeans, 'kmeans++', listData)



doPaddedKmeans()

