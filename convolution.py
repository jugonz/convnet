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
    def __init__(self, numFilters, filterDim, nonlinear, nonlinearDeriv):
        self.numFilters = numFilters
        self.filterDim = filterDim

        # one filterDim x filterDim per filter
        self.W = np.random.randn(numFilters, filterDim, filterDim)
        # biases: one for each filter
        self.B = np.zeros(numFilters)

        self.nonlinearFunc = nonlinear
        self.nonlinearDeriv = nonlinearDeriv

    def init_weights(self, weights):
        # Assumes that weights is a numpy datatype.
        assert(self.W.shape == weights.shape)
        self.W = weights

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

                output = self.nonlinearFunc(signal.correlate2d(image, filt, "valid") + b[j])

                # now, save the output
                convolved[i, j] = output

        return convolved

    # back propagation: return the error from this layer for use in the previous layer
    def backward_prop(self, inp, error, learningRate):
        outputDeriv = self.nonlinearDeriv(self.forward_prop(inp))
        newDelta = error * outputDeriv # * is element-wise prod

        # this really is a convolution
        weightUpdate = signal.convolve2d(inp, newDelta, "valid")

        # Update weights (this isn't a matrix multiplication, unfortunately).
        for i in xrange(self.numFilters):
            for j in xrange(self.filterDim):
                self.W[i][j] += learningRate * weightUpdate[i][j]

        # this should be newDelta flipped 90 degrees and a 2d convolution
        return signal.correlate2d(newDelta, self.W, "valid")

