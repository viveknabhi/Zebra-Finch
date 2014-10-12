from scikits.audiolab import wavread
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


def findFileNames():
	folder = './NickLabAdultZebraFinchSongs/o31_1daypre'
	files = []
	for filename in os.listdir(folder):
		path = os.path.join(folder, filename)
		files.append(path)
	return files

def extractMFCC(fileNames):
	dataDict = {}
	for inpFile in fileNames:
		data, fs = wavread(inpFile)[:2]
		ceps = mfcc(data,fs = fs)[0]
		dataDict[inpFile] = ceps

	return dataDict

	pickle.dump(dataDict,open("mfccDump.p","wb"))

def pickleData(data,writeFile):
	pickle.dump(data,open(writeFile,"wb"))


def loadDataFromWav(fileName):
	files = findFileNames()
	dataDict = extractMFCC(files)
	dataDict = padZeroes(dataDict)
	#dataDict = padNans(dataDict)
	pickleData(dataDict,fileName)
	return dataDict

def padZeroes(dataDict):
	maxLen = max([ceps.shape[0] for ceps in dataDict.values()])
	for key in dataDict:
		rows = maxLen - dataDict[key].shape[0]
		pad = np.zeros((rows,13))
		dataDict[key] = np.vstack((dataDict[key], pad))

	return dataDict


def padNans(dataDict):
	maxLen = max([ceps.shape[0] for ceps in dataDict.values()])
	for key in dataDict:
		rows = maxLen - dataDict[key].shape[0]
		pad = np.empty((rows,13))
		pad[:] = np.NAN
		dataDict[key] = np.vstack((dataDict[key], pad))

	return dataDict

def loadDataFromPickle(fileName):
	data = pickle.load(open(fileName,"rb"))
	return data

def plotCeps(ceps):
	plt.plot(ceps)
	plt.show()


if __name__ == "__main__":
	#To load the data again from the files
	dataDict = loadDataFromWav("mfccDumpPadded.p")
	#dataDict = loadDataFromPickle("mfccDumpPadded.p")
	# for key in dataDict:
	# 	print dataDict[key].shape
		#raw_input()