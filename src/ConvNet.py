from layer import Layer
from convolution_full import ConvolutionLayer
from pooling import PoolingLayer
from fully_connected_batch import FullyConnectedLayer
import numpy as np
import ConfigParser as cp
import json
import misc
import re

class ConvNet:
    def __init__(self, pathToConfigFile):
        self.pathToConfigFile = pathToConfigFile
        self.config = cp.ConfigParser()
        self.config.read(self.pathToConfigFile)

        self.nonlinearFunc = {'tanh': self._tanh, 'sigmoid': self._sigmoid}
        self.nonlinearDeriv = {'tanh': self._tanhDeriv, 'sigmoid': self._sigmoidDeriv}

        self.layers = []
        self.labelSet = {}

        sections = self.config.sections()
        for idx, label in enumerate(self.config.get('Parameters', 'labelSet').split(' ')):
            self.labelSet[label] = idx

        self.learningRate = float(self.config.get('Parameters', 'learningRate'))
        self.momentum = float(self.config.get('Parameters', 'momentum'))

        self.inputLayerPattern = 'InputLayer'
        self.convLayerPattern = 'ConvLayer[0-9]+'
        self.poolLayerPattern = 'PoolLayer[0-9]+'
        self.fullLayerPattern = 'FullLayer[0-9]+'

        # check correct structure
        assert(bool(re.match(self.inputLayerPattern + '(' + self.convLayerPattern + self.poolLayerPattern +\
             ')+' + '(' + self.fullLayerPattern + ')+', reduce(lambda x, y: x + y, sections[1:]))))
        
        for idx, section in enumerate(sections[1:]):
            if bool(re.match(self.inputLayerPattern, section)):
                numChannels = int(self.config.get(section, 'numChannels'))
                channelDim = int(self.config.get(section, 'imgSize'))

            elif bool(re.match(self.convLayerPattern, section)):
                numFilters = int(self.config.get(section, 'numFilters'))
                filterDim = int(self.config.get(section, 'filterDim'))
                type = self.config.get(section, 'type')

                layer = ConvolutionLayer(numChannels, numFilters, filterDim, type)
                if self.config.get(section, 'weights') != 'None':
                    weights = np.array(json.loads(self.config.get(section, 'weights')))
                    weights = np.reshape(weights, (numFilters, numChannels, filterDim, filterDim))

                    biases = np.array(json.loads(self.config.get(section, 'biases')))

                    layer.init_weights(weights, biases)

                self.layers.append(layer)

                numChannels = numFilters
                channelDim = channelDim - filterDim + 1

            elif bool(re.match(self.poolLayerPattern, section)):
                winSize = int(self.config.get(section, 'winSize'))
                type = self.config.get(section, 'type')

                if bool(re.match(self.convLayerPattern, sections[idx+2])):
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

                layer = FullyConnectedLayer(numIn, numOut, self.nonlinearFunc[type],\
                    self.nonlinearDeriv[type])

                if self.config.get(section, 'weights') != 'None':
                    weights = np.array(json.loads(self.config.get(section, 'weights')))
                    biases = np.array(json.loads(self.config.get(section, 'biases')))

                    weights = np.vstack((np.reshape(weights, (numIn-1, numOut)), biases))

                    layer.init_weights(weights)

                self.layers.append(layer)

    def _transformInput(self, imageDim, sample):
        # input sample is (imgDim x imgDim)
        # need to change it to be (1 x imgDim x imgDim) for 1 channel input
        inp = np.zeros((1, imageDim, imageDim))
        inp[0] = sample
        return inp

    def testSample(self, sample, label):
        # First, transform our input to go through the input layer.
        inp = self._transformInput(sample.shape[0], sample)

        # Now, propagate our sample through the network.
        return self.forward_prop(inp)

    # on a pre-trained network, get the outputs for a test set
    def testSet(self, testSet, labels):
        assert len(testSet) == len(labels), "Need 1 label per test image"

        numCorrect = 0
        for i in xrange(len(testSet)):
            # Propagate our sample through the network.
            expectedOutputLabel = labels[i]
            outputVector = self.testSample(testSet[i], labels[i])

            # We have a vector of outputs.
            # Get the index of the max score of this output.
            outputIndex = outputVector.argmax()
            outputLabel = self.labelSet[outputIndex]

            # Save the number of correct labels.
            if outputLabel == expectedOutputLabel:
                numCorrect += 1

        # Report accuracy on this dataset.
        accuracy = numCorrect / float(len(testSet))
        print "Accuracy on this test set was: ", accuracy


    def trainSample(self, sample, label):
        assert len(sample.shape) == 2, "Not a 2D image."
        assert sample.shape[0] == sample.shape[1], "Not a square image."

        # given a config already
        # calls forward_prop and backward_prop on layers
        # forward_prop returns the result of the convolution
        # backward_prop defines how W/B are updated and does the update itself
        # backward_prop takes in a matrix of deltas from the next layer
        # backward_prop returns the deltas (a matrix) from that layer

        # the delta from the last layer is determined by a dictionary
        # self.labelSet that converts labels to the index of the output
        # neuron that is 1 (all others are 0)
        desired = np.array([0.]*len(self.labelSet))
        desired[self.labelSet[label]] = 1.
        desired = np.array([desired])

        # First, transform our input to go through the input layer.
        inp = self._transformInput(sample.shape[0], sample)

        # Now, propagate our sample through the network.
        output = self.forward_prop(inp)

        # Compute the error.
        error = self._error(output, desired)

        # Propagate the error back through the network and update weights.
        self.backward_prop(error)

    def trainSet(self, trainSet, labels, maxEpochs, epochsPerSave):
        # train all samples in loop (multiple times)

        numSamples = trainSet.shape[0]
        for epoch in xrange(maxEpochs):
            print "Epoch: ", epoch
            sample_idxs = np.random.permutation(numSamples)
            for sample_idx in sample_idxs:
                self.trainSample(trainSet[sample_idx,:,:], labels[sample_idx])

            # save trained cnn at this stage
            if epoch%epochsPerSave == 0:
                self._saveTrainedConfigFile(epoch)

        # save final trained cnn
        if maxEpochs%epochsPerSave != 0:
            self._saveTrainedConfigFile(maxEpochs)

    def forward_prop(self, inp):
        # Propagate our sample through each layer of the network.
        output = inp
        for layer in self.layers:
            print "FLayer: ", layer
            output = layer.forward_prop(output)
            print output.shape
        return output

    def backward_prop(self, error):
        # Propagate error through the network.
        i = 0
        for layer in reversed(self.layers):
            print "BLayer: ", layer
            error = layer.backward_prop(error, self.learningRate, self.momentum)
            print error.shape

    # implements the derivate of the error function we're using.
    def _error(self, output, desired):
        return np.subtract(output, desired)

    def saveFilters(self, epochNum):
        # save all the weight matrices to matlab cell array
        numLayers = len(self.layers)
        valuesToSave = {"numLayers" : numLayers} # so matlab can easily know size of array

        filters = np.zeros((numLayers, ), dtype = np.object) # object == cell array
        for i in xrange(numLayers):
            filters[i] = self.layers[i].W
        valuesToSave["filters"] = filters

        # write to disk
        misc.saveToMatlab('filters-' + str(epochNum), valuesToSave)

    def _saveTrainedConfigFile(self, numEpochs):
        pathToTrainedConfigFile = self.pathToConfigFile[:-4] + '-trained-' + str(numEpochs) + '.ini'
        trainedConfigFile = open(pathToTrainedConfigFile,'w')

        sections = self.config.sections()
        for idx,layer in enumerate(self.layers):
            if bool(re.match(self.convLayerPattern, sections[idx+2])):
                numIn = layer.fullLayer.numIn
                numOut = layer.fullLayer.numOut

                self.config.set(sections[idx+2], 'weights', str(layer.fullLayer.W[0:numIn-1,:].flatten('F').tolist()))

                self.config.set(sections[idx+2], 'biases', str(layer.fullLayer.W[numIn-1,:].flatten('F').tolist()))

            elif bool(re.match(self.fullLayerPattern, sections[idx+2])):
                numIn = layer.numIn
                numOut = layer.numOut

                self.config.set(sections[idx+2], 'weights', str(layer.W[0:numIn-1,:].flatten().tolist()))

                self.config.set(sections[idx+2], 'biases', str(layer.W[numIn-1,:].flatten().tolist()))

        self.config.write(trainedConfigFile)
        trainedConfigFile.close()

    def _tanh(self, x):
        return np.tanh(x)

    def _sigmoid(self, x):
        pass

    def _tanhDeriv(self, x):
        return 1.0 - x**2

    def _sigmoidDeriv(self, x):
        pass
