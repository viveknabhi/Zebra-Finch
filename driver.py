import mean_shift_cluster as msc
import readData
import numpy as np
import h_clustering as hc
import kMeans as km
import GMM as gmm
from sklearn.decomposition import PCA
import DBSCAN as dbsc
from collections import defaultdict
import pickle



time = defaultdict(list)
score = defaultdict(list)

timeAvg = defaultdict(list)
scoreAvg = defaultdict(list)


def doPadded():
	data = readData.loadDataFromPickle('mfccDumpPadded.p')
	listData = []
	for item in data:
		listData.append(data[item].flatten())

	listData = np.array(listData)
	print listData.shape
	listData = PCA(n_components = 40).fit_transform(listData)
	print listData.shape
	for i in xrange(2,10,1):
		result =  msc.mean_shift(listData,'euclidean')
		print result
		time['ms'].append(result[2])
		score['ms'].append(result[1])
		result =   hc.Hierarchical_Cluster(listData,i,'euclidean')
		print result
		time['h'].append(result[2])
		score['h'].append(result[1])
		result =   km.euclidian_k_means(listData,i,'euclidean')
		print result
		time['km'].append(result[2])
		score['km'].append(result[1])
		result =   gmm.estimate_GMM(listData,i,'euclidean')
		print result
		time['em'].append(result[2])
		score['em'].append(result[1])
		# result =   dbsc.doDBSCAN(listData)
		# print result
		result =  km.dtw_k_means(listData,i,'euclidean')
		print result
		time['dtw'].append(result[2])
		score['dtw'].append(result[1])


def doAvgCoeff():
	data = readData.loadDataFromPickle('mfccDump.p')
	listData = readData.getAveragedCepstralCoefficients(data)
	print listData.shape
	for i in xrange(2,10,1):
		result =   msc.mean_shift(listData,'euclidean')
		print result
		timeAvg['ms'].append(result[2])
		scoreAvg['ms'].append(result[1])
		result =   hc.Hierarchical_Cluster(listData,i,'euclidean')
		print result
		timeAvg['h'].append(result[2])
		scoreAvg['h'].append(result[1])
		result =   km.euclidian_k_means(listData,i,'euclidean')
		print result
		timeAvg['km'].append(result[2])
		scoreAvg['km'].append(result[1])
		result =   gmm.estimate_GMM(listData,i,'euclidean')
		print result
		timeAvg['em'].append(result[2])
		scoreAvg['em'].append(result[1])
		#result =   dbsc.doDBSCAN(listData)
		result =   km.dtw_k_means(listData,i,'euclidean')
		print result
		timeAvg['dtw'].append(result[2])
		scoreAvg['dtw'].append(result[1])



#doDirectPCA()
doPadded()
print "*************************************************"
doAvgCoeff()


pickle.dump(time,open('timePad.p',"wb"))
pickle.dump(score,open('scorePad.p',"wb"))
pickle.dump(timeAvg,open('timeAvg',"wb"))
pickle.dump(scoreAvg,open('scoreAvg',"wb"))