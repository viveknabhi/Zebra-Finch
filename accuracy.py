import sklearn.metrics as metrics


def getSilhoutteScore(data,labels,sampleSize,metric):
	return  metrics.silhouette_score(data, labels,metric=metric,sample_size=len(data))


def getAccuracy(data,labels,sampleSize,metric):
	return getSilhoutteScore(data,labels,sampleSize,metric)