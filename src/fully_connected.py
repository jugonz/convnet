from layer import Layer
import numpy as np

# FullyConnectedLayer models a traditional
# neural network layer (like a hidden or output layer).
# It includes no bias.
class FullyConnectedLayer(Layer):
    def __init__(self, numIn, numOut, nonlinear, nonlinearDeriv):
        self.numIn = numIn
        self.numOut = numOut
        self.W = np.random.randn(self.numIn, self.numOut)
        self.nonlinearFunc = nonlinear
        self.nonlinearDeriv = nonlinearDeriv

    def init_weights(self, weights):
        # Assumes that weights is a numpy datatype.
        assert(self.W.shape == weights.shape)
        self.W = weights

    # forward_prop(input) computes the activation of the nodes
    # in this layer given outputs from the previous layer.
    def forward_prop(self, inp):
        return self.nonlinearFunc(np.dot(inp, self.W))

    # backward_prop(input, error, learningRate) computes the change
    # to this layer's weights given the current sample,
    # the error (essentially, the delta from the next layer),
    # and the learning rate.
    def backward_prop(self, inp, error, learningRate):
        outputDeriv = self.nonlinearDeriv(self.forward_prop(inp))
        newDelta = outputDeriv * error

        # Update weights (this isn't a matrix multiplication, unfortunately).
        for i in xrange(self.numIn):
            for j in xrange(self.numOut):
                self.W[i][j] += learningRate * newDelta[j] * inp[i]

        return newDelta

