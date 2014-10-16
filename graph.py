import matplotlib.pyplot as plt
import pickle
import math

def getLegend(key):
	if key == 'km':
		return 'K-Means'
	elif key == 'em':
		return 'Expectation Maximization'
	elif key == 'h':
		return 'Hierarchical'
	elif key == 'dtw':
		return 'K-Means with DTW'

dataVar = pickle.load(open('timePad.p','rb'))
K = [i for i in xrange(2,10)]

for key in dataVar:
	if not key == 'ms':
		legend = getLegend(key)
		print legend
		plt.plot(K,dataVar[key],label=legend,linewidth=2)


plt.xlabel('Number of Clusters')
plt.ylabel('Time (s)')
plt.ylim([0.0,10])
plt.legend()
plt.show()
print dataVar


dataVar = pickle.load(open('timeAvg','rb'))
K = [i for i in xrange(2,10)]

for key in dataVar:
	if not key == 'ms':
		legend = getLegend(key)
		print legend
		plt.plot(K,dataVar[key],label=legend,linewidth=2)


plt.xlabel('Number of Clusters')
plt.ylabel('Time (s)')
plt.ylim([0.0,1])
plt.legend()
plt.show()
print dataVar


dataVar = pickle.load(open('scorePad.p','rb'))
K = [i for i in xrange(2,10)]

for key in dataVar:
	if not key == 'ms' and not key == 'dtw':
		legend = getLegend(key)
		print legend
		plt.plot(K,dataVar[key],label=legend,linewidth=2)


plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()
print dataVar

dataVar = pickle.load(open('scoreAvg','rb'))
K = [i for i in xrange(2,10)]

for key in dataVar:
	if not key == 'ms' and not key == 'dtw':
		legend = getLegend(key)
		print legend
		plt.plot(K,dataVar[key],label=legend,linewidth=2)


plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()
print dataVar

