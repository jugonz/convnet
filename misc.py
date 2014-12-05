import numpy as np
import scipy.misc as smisc
import scipy.io as sio

def loadImage(filePath):
    return smisc.imread(filePath)

def saveImage(image, filename):
    smisc.imsave(filename, image)

# Save a dictionary of names and np arrays
# to a .mat file in the current folder.
def saveToMatlab(filename, dictionary):
    sio.savemat(filename, dictionary, oned_as='column')

# Return a dictionary of matlab data.
def readFromMatlab(filename):
    return sio.loadmat(filename)