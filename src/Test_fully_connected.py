from fully_connected import FullyConnectedLayer
from input_layer import InputLayer
import numpy as np

# Tuning paramaters.
learningRate = 0.5
numIterations = 8000
numInput = 2
numHidden = 2
numOutput = 1

# Test samples.
# Here, we're testing XOR.
# if in0 = 0 and in1 = 0 -> out = 0
# if in0 = 1 and in1 = 0 -> out = 1
# if in0 = 0 and in1 = 1 -> out = 1
# if in0 = 1 and in1 = 1 -> out = 0
samples = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
samplesToProc = [np.array(sample) for sample in samples]

# Simple error function to see how much we're improving on each iteration.
def Error(expect, actual):
    val = 0.5 * (expect - actual)**2
    return val

# Nonlinear function and its derivative.
def nonlinear(x):
    #return 1 / (1 + np.exp(-1 * x)) THIS DOESN'T WORK
    return np.tanh(x)

def nonlinearDeriv(x):
    return 1.0 - x**2

# Simple test that runs a sample network to get the output of the XOR function.
def FullyConnectedLayerTest():
    inputLayer = InputLayer(numInput + 1) # The +1 is the bias unit.
    hiddenLayer = FullyConnectedLayer(numInput + 1, numHidden, nonlinear, nonlinearDeriv)
    outputLayer = FullyConnectedLayer(numHidden, numOutput, nonlinear, nonlinearDeriv)

    # Until we're done,
    for i in xrange(numIterations):
        # go through our samples:
        for sample in samplesToProc:
            inputs = sample[0:2]

            # Do forward propagation on the sample.
            inputOut = inputLayer.forward_prop(inputs)
            hiddenOut = hiddenLayer.forward_prop(inputOut)
            outOut = outputLayer.forward_prop(hiddenOut)

            # Now, do backward propagation.
            # Note that computation of weight deltas requires information outside of a layer.
            outWeights = outputLayer.W
            outDelta = sample[2] - outOut
            outBack = outputLayer.backward_prop(hiddenOut, outDelta, learningRate)

            hiddenDelta = np.dot(outWeights, outBack) # Needs weights of output layer BEFORE update!
            hiddenBack = hiddenLayer.backward_prop(inputOut, hiddenDelta, learningRate)

            # Print the current error we're experiencing.
            if i % 100 == 0:
                print "error on iteration %d: %0.3f" % (i, Error(sample[2], outOut)[0])

    # Print our final values.
    print "FINAL VALUES:"
    for sample in samplesToProc:
        inputs = sample[0:2]
        print "input: ", inputs

        # Get the final value.
        inputOut = inputLayer.forward_prop(inputs)
        hiddenOut = hiddenLayer.forward_prop(inputOut)
        outOut = outputLayer.forward_prop(hiddenOut)

        # Print final along with expected, and error.
        print "output: ", outOut
        print "expected output: ", sample[2]
        print "error: %0.2f" % Error(sample[2], outOut)[0]

if __name__ == "__main__":
    FullyConnectedLayerTest()
