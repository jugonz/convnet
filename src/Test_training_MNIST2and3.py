from ConvNet import ConvNet
import extract_mnist
import numpy as np
import unittest

class MNISTConvNetTrainingTest(unittest.TestCase):
	def setUp(self):
		configPath = "MNIST2and3Config.ini"
		self.net = ConvNet(configPath)

		# import just images of 2s and 3s for now
		self.imagesOf2s = extract_mnist.getMNISTTrainingSamplesNum(2)
		self.imagesOf3s = extract_mnist.getMNISTTrainingSamplesNum(3)

	def testTrain2sAnd3s(self):
		numImages = 125 # number of images of each category

		# combine two data sets
		twos = self.imagesOf2s[0:numImages]
		threes = self.imagesOf3s[0:numImages]
		data = np.concatenate((twos, threes))

		# set up labels
		labels = []
		for i in xrange(numImages):
			labels.append("two")
		for i in xrange(numImages):
			labels.append("three")

		# epochs
		maxEpochs = 10
		epochsPerSave = 10

		# run test
		self.net.trainSet(data, labels, maxEpochs, epochsPerSave)
		self.net.testSet(data, labels)
		self.net.saveFilters(maxEpochs)

if __name__ == "__main__":
    unittest.main()
