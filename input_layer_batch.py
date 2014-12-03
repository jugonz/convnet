from layer import Layer
import numpy as np

# An InputLayer is models an input layer with
# with a constant bias node.
class InputLayer(Layer):
    def __init__(self, numIn, numOut = 0, nonLinear = None, nonlinearDeriv = None, bias = 1):
        self.numIn = numIn - 1
        self.bias = bias

    # For an InputLayer, our activation is simply our input
    # as well as the output of our extra bias node.
    def forward_prop(self, inp):
        # assume inp is np array (assume 2-D)
        result = np.zeros([len(inp), self.numIn + 1])
        for i in xrange(len(inp)):
            result[i] = np.append(inp[i], self.bias)
        return result

    # We do no training on InputLayers.
    def backward_prop(self, inp, error, learningRate):
        return None