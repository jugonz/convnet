from layer import Layer
import numpy as np

# We generate our own W (filters) and B (biases):
# W randomly (using the normal distribution)
# B is all zeros
class FullyConnectedLayer(Layer):
    def __init__(self, numIn, numOut, learningRate):
        # need to initialize weights/biases
        self.W = np.random.randn(numIn, numOut)
        self.B = np.zeros(numOut)
        self.learningRate = learningRate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def forward_prop(self, inp):
        # for now, assume inp is np datatype
        # this can be used for large data
        return self._sigmoid(np.dot(inp, self.W) + self.B)

    # back propagation: return the delta from this layer for use in the previous layer
    # works for all layers except output layer
    def backward_prop(self, deltaNextLayer):
        output = self.forward_prop(inp)
        outputDeriv = output * (1 - output)

        return np.dot(self.W.T, deltaNextLayer) * outputDeriv # * is element-wise prod

    def update(self, w_grad, b_grad):
        pass