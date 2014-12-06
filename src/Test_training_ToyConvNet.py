from ConvNet import ConvNet
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

	def testTrain1Image(self):
		data = np.zeros((1, self.imageDim, self.imageDim))
		data[0] = np.array([self.images[0]])
		maxEpochs = 100
		epochsPerSave = 101 # no saving in this test

		self.net.trainSet(data, self.labels, maxEpochs, epochsPerSave)

#	def testTrain6Images(self):
#		data = self.images
#		maxEpochs = 1000
#		epochsPerSave = 10001 # no saving in this test either

#		self.net.trainSet(data, self.labels, maxEpochs, epochsPerSave)

#	def testTrain6ImagesAndTest(self):
#		data = self.images
#		maxEpochs = 1000
#		epochsPerSave = 10001 # no saving in this test either :)

#		self.net.trainSet(data, self.labels, maxEpochs, epochsPerSave)
#		self.net.testSet(data, self.labels)


if __name__ == "__main__":
    unittest.main()
