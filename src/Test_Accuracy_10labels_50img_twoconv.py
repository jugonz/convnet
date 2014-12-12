from ConvNet import ConvNet
import extract_mnist
import read_mnist
import numpy as np
import unittest
import misc

class MNISTConvNetTestingTest(unittest.TestCase):
    def setUp(self):
        configPath_prefix = "../10labels_50img_twoconv_trained/MNIST10labels_50img_twoConv"
        self.nets = [ConvNet(configPath_prefix+".ini")]
        for i in xrange(5,55,5):
            self.nets.append(ConvNet(configPath_prefix + "-trained-" + str(i) + ".ini"))

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
        
		# run test
        for net in self.nets:
            net.testSet(self.images, str_labels)

if __name__ == "__main__":
    unittest.main()
