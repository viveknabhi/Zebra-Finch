from scikits.audiolab import wavread
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt

# data, fs = wavread('Audio1.wav')[:2]

# ceps = mfcc(data, fs=fs)[0]
# plt.plot(ceps)
# plt.show()


def findFileNames():
	return fileNames

def main():
	files = findFileNames()

if __name__ == "__main__":
	main