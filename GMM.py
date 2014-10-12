import readData
import numpy as np
from sklearn.mixture import GMM

from sklearn.decomposition import PCA
from time import time
import sklearn.metrics as metrics
import accuracy

def estimate_GMM(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    labels = estimator.predict(data)
    print len(labels)
    print 'name', 'time' ,'silhouette_score' 
    print('% 9s   %.2fs   %.3f '
          % (name, (time() - t0),accuracy.getAccuracy(data,labels,len(data),'cosine')))



def doPaddedGMM():
	data = readData.loadDataFromPickle('mfccDumpPadded.p')
	listData = []
	for item in data:
		listData.append(data[item].flatten())

	listData = np.array(listData)
	for i in xrange(4,10,1):
		listData = PCA(n_components=i).fit_transform(listData)
		gmm = GMM(n_components=i,n_iter = 5,n_init =1)
		estimate_GMM(gmm, 'gmm', listData)


def doAvgCoeffGMM():
	data = readData.loadDataFromPickle('mfccDump.p')
	listData = readData.getAveragedCepstralCoefficients(data)
	for i in xrange(4,10,1):
		listData = PCA(n_components=i).fit_transform(listData)
		gmm = GMM(n_components=i,n_iter = 5,n_init =1)
		estimate_GMM(gmm, 'gmm', listData)


doPaddedGMM()
doAvgCoeffGMM()