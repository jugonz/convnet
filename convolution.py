from layer import Layer
import numpy as np
import scipy.signal as signal

# What do we need to perform forward propagation?
#
# filterDim - N in (N x N) dimensions of filters
# numFilters - number of filters
# input - input to convolve (can get dimensions from numpy arrays)
#
# We generate our own W (filters) and B (biases)
class ConvolutionLayer(Layer):
    def __init__(self, numFilters, filterDim, learningRate):
        self.numFilters = numFilters
        self.filterDim = filterDim

        # need to initialize weights... one filterDim x filterDim per filter
        # in numpy indexing works differently than matlab, so store
        # number of filter FIRST
        self.W = np.random.randn(numFilters, filterDim, filterDim)
        # biases: one for each filter
        self.B = np.zeros(numFilters)
        self.learningRate = learningRate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def forward_prop(self, inp):
        # for now, assume inp is np datatype
        # and size is accessible via .shape attribute
        # and size is NumberOfImages x N x N (square input)
        numImages = inp.shape[0]
        convSize = inp.shape[1] - filterDim + 1

        convolved = np.zeros(numImages, self.numFilters, convSize, convSize)
        # unoptimized
        for i in xrange(numImages):
            for j in xrange(self.numFilters):
                # instead of doing the 2-d convolution with the flipped filter,
                # do the 2-d cross-correlation to skip flipping
                image = inp[i]
                filt = self.W[j]

                output = self._sigmoid(signal.correlate2d(image, filt, "valid"))

                # now, save the output
                convolved[i, j] = output

        return convolved

    # back propagation: return the delta from this layer for use in the previous layer
    # works for all layers except output layer
    def backward_prop(self, deltaNextLayer):
        # deltaNextLayer is the delta from the next layer (closer to the output layer)
        output = self.forward_prop(inp)
        outputDeriv = output * (1 - output)

        return np.dot(self.W.T, deltaNextLayer) * outputDeriv # * is element-wise prod

    def update(self, w_grad, b_grad):
        pass