from ConvNet import ConvNet
import read_mnist
import numpy as np
import unittest
import misc

class MNISTConvNetTestingTest(unittest.TestCase):
	def setUp(self):
		configPath = "MNIST_10_Config-trained-55.ini"
		self.net = ConvNet(configPath)

		# get all data
		(data, labels) = read_mnist.getMNISTTest()
		labels.flags.writeable = True
		self.labels = np.zeros(labels.shape)
		self.data = data
		for i in xrange(len(labels)):
			self.labels[i] = int(labels[i])

	def testTestAll(self):
		# no training here

		# run test
		maxEpochs = 1000
		epochsPerSave = 5

		print "data: ", self.data
		print "labels: ", self.labels

		#self.net.trainSet(data, labels, maxEpochs, epochsPerSave)
		self.net.testSet(self.data, self.labels)
		#self.net.saveFilters(1010)
		#self.net.saveActivations(1010)

if __name__ == "__main__":
    unittest.main()
