from ConvNet import ConvNet
import extract_mnist
import numpy as np
import unittest

class MNISTConvNetTestingTest(unittest.TestCase):
	def setUp(self):
		configPath = "MNIST2and3and7Config_twoConv-trained-100.ini"
		self.net = ConvNet(configPath)

		# import just images of 2s and 3s for now
		self.imagesOf2s = extract_mnist.getMNISTTestSamplesNum(2)
		self.imagesOf3s = extract_mnist.getMNISTTestSamplesNum(3)
		self.imagesOf7s = extract_mnist.getMNISTTestSamplesNum(7)

	def testTest2sAnd3sAnd7s(self):
		numImages = 1000 # number of images of each category

		# combine two data sets
		twos = self.imagesOf2s[0:numImages]
		threes = self.imagesOf3s[0:numImages]
		sevens = self.imagesOf7s[0:numImages]
		data = np.concatenate((twos, threes, sevens))

		# set up labels
		labels = []
		for i in xrange(numImages):
			labels.append("two")
		for i in xrange(numImages):
			labels.append("three")
		for i in xrange(numImages):
			labels.append("seven")

		# run test
		self.net.testSet(data, labels)
		self.net.saveFilters(100)

if __name__ == "__main__":
    unittest.main()
