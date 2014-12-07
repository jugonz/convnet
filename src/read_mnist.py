import numpy as np

## File paths.
testImgsLoc = "../test_imgs/mnist/t10k-images-idx3-ubyte"
trainImgsLoc = "../test_imgs/mnist/train-images-idx3-ubyte"
testLblsLoc = "../test_imgs/mnist/t10k-labels-idx1-ubyte"
trainLblsLoc = "../test_imgs/mnist/train-labels-idx1-ubyte"

## Magic numbers. These are used to verify file contents (don't change).
imgsMagic = 2051
lblsMagic = 2049

##
## Utility functions (call these!).
##

# Returns a tuple [images, labels] of training data.
def getMNISTTraining():
    return [readMNISTImages(trainImgsLoc), readMNISTLabels(trainLblsLoc)]

def getMNISTTest():
    return [readMNISTImages(testImgsLoc), readMNISTLabels(testLblsLoc)]

##
## MNIST reading functions.
##

# Read MNIST label file (training or test).
def readMNISTLabels(location):
    mnist = open(location, 'rb') # Read, binary mode.

    # MNIST files start with a magic number.
    # Throw an error if we don't find the expected file.
    firstByte = readInt32(mnist)
    assert firstByte == lblsMagic, "File read was NOT MNIST file!"

    # The next 4 bytes in an MNIST image file is the number of labels.
    numImages = readInt32(mnist)

    # Store labels in an np array.
    # For each image, save its label (number from 0-9).
    labels = readRow(mnist, numImages)

    # Make sure we read the whole file.
    if mnist.read(1) != "":
        print "File was longer than expected!"

    return labels

# Read MNIST image file (training or test).
def readMNISTImages(location):
    mnist = open(location, 'rb') # Read, binary mode.

    # MNIST files start with a magic number.
    # Throw an error if we don't find the expected file.
    firstByte = readInt32(mnist)
    assert firstByte == imgsMagic, "File read was NOT MNIST file!"

    # The next 12 bytes in an MNIST image file are parameters.
    numImages = readInt32(mnist)
    numRows = readInt32(mnist)
    numCols = readInt32(mnist)

    # Store images in a (numImages x numRows x numCols) np array.
    imgs = np.zeros((numImages, numRows, numCols))

    # For each image, get our pixels and save it in imgs.
    for imgNum in xrange(numImages):
        imgData = np.zeros((numRows, numCols))
        # Each byte in the file is a new pixel of the image, row-wise.
        for i in xrange(numRows):
            imgData[i] = readRow(mnist, numCols)
        # Save into our np array.
        imgs[imgNum] = imgData

    # Make sure we read the whole file.
    if mnist.read(1) != "":
        print "File was longer than expected!"

    return imgs

##
## Helper functions for reading from MNIST byte files.
##

def convertToInt(hexData):
    return int(hexData.encode('hex'), 16)

def readInt32(fileObj):
    return convertToInt(fileObj.read(4))

def readRow(fileObj, rowCount):
    # assume each value is 1 byte
    return np.frombuffer(fileObj.read(rowCount), np.uint8)
