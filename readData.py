from scikits.audiolab import wavread
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import pickle
import os


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


def loadDataFromWav():
	files = findFileNames()
	dataDict = extractMFCC(files)
	pickleData(dataDict,"mfccDump.p")
	return dataDict


def loadDataFromPickle(fileName):
	data = pickle.load(open(fileName,"rb"))
	return data

def plotCeps(ceps):
	plt.plot(ceps)
	plt.show()


if __name__ == "__main__":
	dataDict = loadDataFromPickle("mfccDump.p")
	print len(dataDict)