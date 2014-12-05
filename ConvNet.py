from layer import Layer
from convolution_full import ConvolutionLayer
from pooling import PoolingLayer
from fully_connected_batch import FullyConnectedLayer
import numpy as np
import ConfigParser as cp
import json
import re

class ConvNet:
    def __init__(self, pathToConfigFile):
        self.config = cp.ConfigParser()
        self.config.read(pathToConfigFile)

        self.nonlinearFunc = {'tanh': self._tanh, 'sigmoid': self._sigmoid}
        self.nonlinearDeriv = {'tanh': self._tanhDeriv, 'sigmoid': self._sigmoidDeriv}

        self.layers = []
        sections = self.config.sections()

        inputLayerPattern = 'InputLayer'
        convLayerPattern = 'ConvLayer[0-9]+'
        poolLayerPattern = 'PoolLayer[0-9]+'
        fullLayerPattern = 'FullLayer[0-9]+'

        # check correct structure
        assert(bool(re.match(inputLayerPattern + '(' + convLayerPattern + poolLayerPattern +\
             ')+' + '(' + fullLayerPattern + ')+', reduce(lambda x, y: x + y, sections))))

        for idx,section in enumerate(sections):
            if bool(re.match(inputLayerPattern, section)):
                numChannels = int(self.config.get(section, 'numChannels'))
                channelDim = int(self.config.get(section, 'imgSize'))

            elif bool(re.match(convLayerPattern, section)):
                numFilters = int(self.config.get(section, 'numFilters'))
                filterDim = int(self.config.get(section, 'filterDim'))
                type = self.config.get(section, 'type')

                layer = ConvolutionLayer(numChannels, numFilters, filterDim, type)
                if self.config.get(section, 'weights') != 'None':
                    weights = json.loads(self.config.get(section, 'weights'))
                    weights = np.reshape(weights, (numChannels, filterDim, filterDim))

                    biases = json.loads(self.config.get(section, 'biases'))
                    biases = np.array(biases)

                    layer.init_weights(weights, biases)

                self.layers.append(layer)

                numChannels = numFilters
                channelDim = channelDim - filterDim + 1

            elif bool(re.match(poolLayerPattern, section)):
                winSize = int(self.config.get(section, 'winSize'))
                type = self.config.get(section, 'type')

                if bool(re.match(convLayerPattern, sections[idx+1])):
                    nextLayer = 'conv'
                else:
                    nextLayer = 'full'

                layer = PoolingLayer(numChannels, channelDim, winSize, nextLayer, type)

                self.layers.append(layer)

                channelDim = channelDim/winSize
                numOut = (channelDim**2*numChannels)+1


            else:
                numIn = numOut
                numOut = int(self.config.get(section, 'numOut'))
                type = self.config.get(section, 'type')

                layer = FullyConnectedLayer(numIn, numOut, self.nonlinearFunc[type], self.nonlinearDeriv[type])

                if self.config.get(section, 'weights') != 'None':
                    weights = json.loads(self.config.get(section, 'weights'))
                    weights = np.array(weights)

                    biases = json.loads(self.config.get(section, 'biases'))
                    biases = np.array(biases)

                    layer.init_weights(weights, biases)

                self.layers.append(layer)

    def _transformInput(self, imageDim, input):
        # input sample is (imgDim x imgDim)
        # need to change it to be (1 x imgDim x imgDim) for 1 channel input
        inp = np.zeros((1, imageDim, imageDim))
        inp[0] = sample
        return inp

    def testSample(self, sample, label):
        # First, transform our input to go through the input layer.
        inp = self._transformInput(sample.shape[0], sample)

        # Now, propagate our sample through the network.
        output = self.forward_prop(inp)

    def trainSample(self, sample, label):
        sample = np.array(sample)
        assert len(sample.shape) == 2, "Not a 2D image."
        assert sample.shape[0] == sample.shape[1], "Not a square image."
        # given a config already
        # calls forward_prop and backward_prop on layers
        # forward_prop returns the result of the convolution
        # backward_prop defines how W/B are updated and does the update itself
        # backward_prop takes in a matrix of deltas from the next layer
        # backward_prop returns the deltas (a matrix) from that layer
        # the delta from the last layer is determined by a dictionary
        # self.labelDict that converts labels to the index of the output
        # neuron that is 1 (all others are 0)
        self.learningRate = 0.5
        self.momentum = 0.1

        self.labelDict = {str(i):i for i in xrange(11)}
        desired = np.array([0.]*10)
        desired[self.labelDict[label]] = 1.

        # First, transform our input to go through the input layer.
        inp = self._transformInput(sample.shape[0], sample)

        # Now, propagate our sample through the network.
        output = self.forward_prop(inp)

        # Compute the error.
        error = self._error(output, desired)

        # Propagate the error back through the network and update weights.
        self.backward_prop(error)

    def forward_prop(self, inp):
        # Propagate our sample through each layer of the network.
        output = inp
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

    def backward_prop(self, error):
        # Propagate error through the network.
        for layer in reversed(self.layers):
            error = layer.backward_prop(error, learningRate, momentum)

    # implements the derivate of the error function we're using.
    def _error(self, output, desired):
        return np.subtract(output, desired)

    def trainSet(self, trainSet, labels, maxEpochs):
        # train all samples in loop (multiple times)
        pass

    def _tanh(self, x):
        return np.tanh(x)

    def _sigmoid(self, x):
        pass

    def _tanhDeriv(self, x):
        return 1.0 - x**2

    def _sigmoidDeriv(self, x):
        pass
