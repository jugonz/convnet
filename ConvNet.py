from layer import Layer
from convolution_full import ConvolutionLayer
from pooling import PoolingLayer
import numpy as np
import ConfigParser as cp
import json as js
import re

class ConvNet:
    def __init__(self, pathToConfigFile):
        self.config = cp.ConfigParser()
        self.config.read(pathToConfigFile)
        
        self.layers = []
        sections = self.config.sections()
        
        inputLayerPattern = 'InputLayer'
        convLayerPattern = 'ConvLayer[0-9]+'
        poolLayerPattern = 'PoolLayer[0-9]+'
        fullLayerPattern = 'FullLayer[0-9]+'
        
        # check correct structure
        assert(bool(re.match(inputLayerPattern + '[' + convLayerPattern + poolLayerPattern + ']+' + '[' + fullLayerPattern + ']+', reduce(lambda x, y: x + y, a))))
        
        for section in sections:
            if bool(re.match(inputLayerPattern, section)):
                numChannels = int(self.config.get(section, 'numChannels'))
                channelDim = int(self.config.get(section, 'imgSize'))
            elif bool(re.match(convLayerPattern, section)):
                numFilters = int(self.config.get(section, 'numFilters'))
                filterDim = int(self.config.get(section, 'filterDim'))
                type = self.config.get(section, 'type')
                
                convLayer = ConvolutionLayer(numChannels, numFilters, filterDim, type)
                if self.config.get(section, 'weights') != 'None':
                    pass
                
                numChannels = numFilters
                channelDim = channelDim - filterDim + 1
            elif bool(re.match(fullLayerPattern, section)):
                pass
            else:
                pass

    def trainSample(self, sample, label):
        # given a config already
        # calls forward_prop and backward_prop on layers
        # forward_prop returns the result of the convolution
        # backward_prop defines how W/B are updated and does the update itself
        # backward_prop takes in a matrix of deltas from the next layer
        # backward_prop returns the deltas (a matrix) from that layer
        pass

    def trainSet(self, trainSet, labels, maxEpochs):
        # train all samples in loop (multiple times)
        pass

    # define nonlinear funcs and derivatives here
