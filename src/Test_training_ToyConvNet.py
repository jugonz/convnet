from ConvNet import ConvNet
import extract_mnist
import misc
import numpy as np
import unittest

class ToyConvNetTrainingTest(unittest.TestCase):
	def setUp(self):
		configPath = "MultipleConvPoolLayersConfig.ini"
		imagePathPrefix = "../test_imgs/lines/"
		imagePaths = ["horiz_line0.png", "horiz_line1.png", "horiz_line2.png",
			"vert_line0.png", "vert_line1.png", "vert_line2.png"]
		self.labels = np.array(["horiz", "horiz", "horiz", "vert", "vert", "vert"])
		self.imageDim = 32
		self.numImages = 6
		assert len(self.labels) == self.numImages, "Not enough labels to run!"
		assert len(imagePaths) == self.numImages, "Not enough images to run!"

		self.net = ConvNet(configPath)
		self.images = np.zeros((self.numImages, self.imageDim, self.imageDim))
		for i in xrange(self.numImages):
			self.images[i] = misc.loadImage(imagePathPrefix + imagePaths[i])

	# def testTrain1Image(self):
	# 	data = np.zeros((1, self.imageDim, self.imageDim))
	# 	data[0] = np.array([self.images[0]])
	# 	maxEpochs = 100
	# 	epochsPerSave = 101 # no saving in this test

	# 	self.net.trainSet(data, np.array([self.labels[0]]), maxEpochs, epochsPerSave)
	# 	self.net.testSet(data, np.array([self.labels[0]]))

	# def testTrain6Images(self):
	# 	data = self.images
	# 	maxEpochs = 1000
	# 	epochsPerSave = 10001 # no saving in this test either

	# 	self.net.trainSet(data, self.labels, maxEpochs, epochsPerSave)

	def testTrain6ImagesAndTest(self):
		data = self.images
		maxEpochs = 500
		epochsPerSave = 500 # save at the end of this test

		self.net.trainSet(data, self.labels, maxEpochs, epochsPerSave)
		self.net.testSet(data, self.labels)
		self.net.saveFilters(maxEpochs) # for visualization

	# To run this test, change the configPath in setUp() to point to a
	# pretrained ConvNet config file.
	# def testPrintTrainedFilters(self):
	# 	data = self.images

	# 	self.net.testSet(data, self.labels)
	# 	self.net.saveFilters(0)


if __name__ == "__main__":
    unittest.main()
