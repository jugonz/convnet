import read_mnist
import numpy as np

def getMNISTTrainingSamplesNum(digit):
	assert digit >=0 and digit <= 9, "Invalid digit passed."

	mnistTraining = read_mnist.getMNISTTraining()
	images = mnistTraining[0]
	labels = mnistTraining[1]

	imagesMatchingDigits = []
	for i in xrange(labels.shape[0]):
		if labels[i] == digit:
			imagesMatchingDigits.append(images[i])

	return np.array(imagesMatchingDigits)

if __name__ == '__main__':
	print getMNISTTrainingSamplesNum(3)[0:125, :, :].shape