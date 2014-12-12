from ConvNet import ConvNet
import extract_mnist
import read_mnist
import numpy as np
import unittest
import misc

class MNISTMisclassifyTest(unittest.TestCase):
    def setUp(self):
        configPath = "../10labels_50img_twoconv_trained/MNIST10labels_50img_twoConv-trained-50.ini"
        self.net = ConvNet(configPath)

        # import just images of 2s and 3s for now
        (self.images, self.labels) = read_mnist.getMNISTTest()
		
    def testTest10Labels(self):
        str_labels = []
        
        for digit in self.labels:
            if digit == 0:
                str_labels.append("zero")
            elif digit == 1:
                str_labels.append("one")
            elif digit == 2:
                str_labels.append("two")
            elif digit == 3:
                str_labels.append("three")
            elif digit == 4:
                str_labels.append("four")
            elif digit == 5:
                str_labels.append("five")
            elif digit == 6:
                str_labels.append("six")
            elif digit == 7:
                str_labels.append("seven")
            elif digit == 8:
                str_labels.append("eight")
            elif digit == 9:
	            str_labels.append("nine")
        
        numWrong = 10
        
		# run test
        for i in xrange(len(self.images)):
            # Propagate our sample through the network.
            expectedOutputLabel = str_labels[i]
            outputVector = self.net.testSample(self.images[i], self.labels[i])
            sortedOutputVector = sorted(outputVector[0])

            # We have a vector of outputs.
            # Get the index of the max score of this output.
            largest = outputVector[0].argmax()
            outputLabel = self.net.labelSetReversed[largest]
            secondLargest = outputVector[0].tolist().index(sortedOutputVector[1])

            # Save the number of correct labels.
            if outputLabel != expectedOutputLabel:
                misc.saveImage(self.images[i], "misclass_"+str(expectedOutputLabel)+"_"+str(largest)+"_"+str(secondLargest)+".png")

if __name__ == "__main__":
    unittest.main()
