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
        self.lastInput = None
        self.lastActivation = None
        self.lastWupdate = np.zeros([self.numIn, self.numOut])

        # From Efficient BackProp: make the standard deviation
        # of the weights be 1/sqrt(num inputs).
        self.W /= np.sqrt(numIn)

    def init_weights(self, weights):
        # Assumes that weights is a numpy datatype.
        assert(self.W.shape == weights.shape)
        self.W = weights

    # forward_prop(input) computes the activation of the nodes
    # in this layer given outputs from the previous layer.
    def forward_prop(self, inp):
        self.lastInput = inp
        self.lastActivation = self.nonlinearFunc(np.dot(inp, self.W))
        return self.lastActivation

    # backward_prop(input, error, learningRate, momentum) computes the change
    # to this layer's weights for the last propagated sample given
    # the error (the delta from the next layer), the learning rate,
    # and the momentum (contribution of previous weight update).
    def backward_prop(self, error, learningRate, momentum):
        assert(self.lastActivation != None)
        assert(self.lastInput != None)
        outputDeriv = self.nonlinearDeriv(self.lastActivation)
        newDelta = outputDeriv * error

        W_update = np.zeros([self.numIn, self.numOut])

        # Update weights (this isn't a matrix multiplication, unfortunately).
        for inpIt in xrange(len(self.lastInput)):
            inp = self.lastInput[inpIt]
            for i in xrange(self.numIn):
                for j in xrange(self.numOut):
                    update = newDelta[inpIt][j] * inp[i]
                    self.W[i][j] += (learningRate * update) +\
                        (momentum * self.lastWupdate[i][j])
                    self.lastWupdate[i][j] = update

        W_update[...] /= len(self.lastInput)
        self.W = np.add(self.W, W_update)

        return newDelta
