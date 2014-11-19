from layer import Layer
import numpy as np

class ConvNet:
    def __init__(self, pathToConfigFile):
        # ...
        # eventually…
        self.layers = []

    def trainSample(self):
        # given a config already
        # calls forward_prop and backward_prop on layers
        # forward_prop returns the result of the convolution
        # backward_prop defines how W/B are updated and does the update itself
        # backward_prop takes in a matrix of deltas from the next layer
        # backward_prop returns the deltas (a matrix) from that layer

    def trainSet(self):
        # train all samples in loop (multiple times)

    # define nonlinear funcs and derivatives here