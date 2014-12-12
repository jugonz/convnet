from ConvNet import ConvNet
import extract_mnist
import numpy as np
import unittest
import misc

class MNISTConvNetTestingTest(unittest.TestCase):
	def setUp(self):
		configPath = "MNIST_10_Config.ini"
		self.net = ConvNet(configPath)

		# import all images
		self.imagesOf0s = extract_mnist.getMNISTTestSamplesNum(0)
		self.imagesOf1s = extract_mnist.getMNISTTestSamplesNum(1)
		self.imagesOf2s = extract_mnist.getMNISTTestSamplesNum(2)
		self.imagesOf3s = extract_mnist.getMNISTTestSamplesNum(3)
		self.imagesOf4s = extract_mnist.getMNISTTestSamplesNum(4)
		self.imagesOf5s = extract_mnist.getMNISTTestSamplesNum(5)
		self.imagesOf6s = extract_mnist.getMNISTTestSamplesNum(6)
		self.imagesOf7s = extract_mnist.getMNISTTestSamplesNum(7)
		self.imagesOf8s = extract_mnist.getMNISTTestSamplesNum(8)
		self.imagesOf9s = extract_mnist.getMNISTTestSamplesNum(9)

	def testTestAll(self):
		numImages = 100 # number of images of each category

		# combine all data sets
		zeros = self.imagesOf0s[0:numImages]
		ones = self.imagesOf1s[0:numImages]
		twos = self.imagesOf2s[0:numImages]
		threes = self.imagesOf3s[0:numImages]
		fours = self.imagesOf4s[0:numImages]
		fives = self.imagesOf5s[0:numImages]
		sixs = self.imagesOf6s[0:numImages]
		sevens = self.imagesOf7s[0:numImages]
		eights = self.imagesOf8s[0:numImages]
		nines = self.imagesOf9s[0:numImages]
		data = np.concatenate((zeros, ones, twos, threes, fours, fives, sixs, \
			sevens, eights, nines))

		# set up labels
		labels = []
		for i in xrange(numImages):
		 	labels.append("zero")
		for i in xrange(numImages):
		    labels.append("one")
		for i in xrange(numImages):
		 	labels.append("two")
		for i in xrange(numImages):
		    labels.append("three")
		for i in xrange(numImages):
		 	labels.append("four")
		for i in xrange(numImages):
		 	labels.append("five")
		for i in xrange(numImages):
		    labels.append("six")
		for i in xrange(numImages):
		 	labels.append("seven")
		for i in xrange(numImages):
		    labels.append("eight")
		for i in xrange(numImages):
		 	labels.append("nine")

		# run test
		maxEpochs = 1000
		epochsPerSave = 5

		self.net.trainSet(data, labels, maxEpochs, epochsPerSave)
		self.net.testSet(data, labels)
		self.net.saveFilters(1010)
		self.net.saveActivations(1010)

if __name__ == "__main__":
    unittest.main()
