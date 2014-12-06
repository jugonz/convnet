from layer import Layer
from fully_connected_batch import FullyConnectedLayer
import numpy as np

class ConvolutionLayer(Layer):
    def __init__(self, numChannels, numFilters, filterDim, type='tanh'):
        self.numChannels = numChannels
        self.numFilters = numFilters
        self.filterDim = filterDim
        self.type = type

        self.nonlinearFunc = {'tanh': self._tanh, 'sigmoid': self._sigmoid}
        self.nonlinearDeriv = {'tanh': self._tanhDeriv, 'sigmoid': self._sigmoidDeriv}

        self.fullLayer = FullyConnectedLayer((self.filterDim**2)*self.numChannels+1,\
        self.numFilters, self.nonlinearFunc[self.type], self.nonlinearDeriv[self.type])

    def init_weights(self, weights, bias):
        # weights.shape = (k, c, w_k, w_k)

        # check same number of filters
        assert(weights.shape[0] == self.numFilters)

        # check bias matches filters
        assert(bias.shape[0] == self.numFilters)

        # check same number of channels
        assert(weights.shape[1] == self.numChannels)

        # check square kernel
        assert(weights.shape[2] == weights.shape[3])

        # check kernel is right size
        assert(weights.shape[2] == self.filterDim)

        numInputs = (self.filterDim**2)*self.numChannels+1

        W = np.zeros((numInputs, self.numFilters))

        for i in xrange(self.numFilters):
            W[0:numInputs-1,i] = weights[i,:,:,:].flatten()

        W[numInputs-1,:] = bias

        self.fullLayer.init_weights(W)

    def forward_prop(self, inp):
        inp = self._transformInput(inp)
        out = self.fullLayer.forward_prop(inp)
        return self._transformOutput(out)

    def backward_prop(self, error, learningRate, momentum):
        error = self._transformInputError(error)
        print error.shape
        newDelta = self.fullLayer.backward_prop(error, learningRate, momentum)
        print error.shape
        return self._transformOutputError(newDelta)

    def _transformInput(self, inp):
        # inp.shape = (c, w_img, w_img)

        # check same number of channels
        #print inp.shape
        #print self.numChannels
        assert(inp.shape[0] == self.numChannels)

        # check square image
        assert(inp.shape[1] == inp.shape[2])

        imgSize = inp.shape[1]
        convSize = imgSize - self.filterDim + 1
        numInputs = (self.filterDim**2)*self.numChannels+1

        transInput = np.zeros((convSize**2, numInputs))

        for j in xrange(convSize):
            for i in xrange(convSize):
                transInput[j*convSize+i,0:numInputs-1] = inp[:,j:j+self.filterDim,\
                    i:i+self.filterDim].flatten()

        transInput[:,numInputs-1] = np.ones(convSize**2)

        return transInput

    def _transformOutput(self, out):
        # check same number of filters
        assert(out.shape[1] == self.numFilters)

        # check square image
        assert(np.sqrt(out.shape[0])%int(np.sqrt(out.shape[0])) == 0)

        imgSize = int(np.sqrt(out.shape[0]))

        transOutput = np.zeros((self.numFilters, imgSize, imgSize))

        for i in xrange(self.numFilters):
            transOutput[i,:,:] = np.reshape(out[:,i], (imgSize, imgSize))

        return transOutput

    def _transformInputError(self, inpError):
        # inpError.shape = (k, w_img - w_k +1, w_img - w_k + 1)

        # check same number of filters
        assert(inpError.shape[0] == self.numFilters)

        # check square image
        assert(inpError.shape[1] == inpError.shape[2])

        imgSize = inpError.shape[1]

        transInputError = np.zeros((imgSize**2, self.numFilters))

        for j in xrange(imgSize):
            for i in xrange(imgSize):
                transInputError[j*imgSize+i,:] = inpError[:,j,i]

        return transInputError

    def _transformOutputError(self, outError):
        # outError.shape = (w_img^2, 1+w_k^2)
        print outError.shape
        # check same number of channels
        #assert(outError.shape[1] == 1+self.filterDim**2)

        # check square image
        assert(np.sqrt(outError.shape[0])%int(np.sqrt(outError.shape[0])) == 0)

        imgSize = int(np.sqrt(outError.shape[0]))

        transOutputError = np.zeros((self.numChannels, imgSize, imgSize))

        for j in xrange(imgSize):
            for i in xrange(imgSize):
                transOutputError[:,j,i] = outError[j*imgSize+i,:]

        return transOutputError

    def _tanh(self, x):
        return np.tanh(x)

    def _sigmoid(self, x):
        pass

    def _tanhDeriv(self, x):
        return 1.0 - x**2

    def _sigmoidDeriv(self, x):
        pass

