from fully_connected_batch import FullyConnectedLayer
from input_layer_batch import InputLayer
import numpy as np

# Tuning paramaters.
learningRate = 0.5
momentum = 0.1
numIterations = 8000
numInput = 2
numHidden = 2
numOutput = 1

# Test samples.
# Here, we're testing XOR in batches.
# if in0 = 0 and in1 = 0 -> out = 0
# if in0 = 1 and in1 = 0 -> out = 1
# if in0 = 0 and in1 = 1 -> out = 1
# if in0 = 1 and in1 = 1 -> out = 0
samples = [[0, 0], [1, 0], [0, 1], [1, 1]]
desired = [np.array([[0], [1], [1], [0]])]
samplesToProc = [[np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]]

# Simple error function to see how much we're improving on each iteration.
def Error(expect, actual):
    val = 0.5 * np.subtract(expect,actual)**2
    return np.sum(val)

# Nonlinear function and its derivative.
def nonlinear(x):
    return np.tanh(x)

def nonlinearDeriv(x):
    return 1.0 - x**2

# Simple test that runs a sample network to get the output of the XOR function.
# Like FullyConnectedLayerTest, but tests using batches.
def FullyConnectedLayerBatchTest():
    inputLayer = InputLayer(numInput + 1) # The +1 is the bias unit.
    hiddenLayer = FullyConnectedLayer(numInput + 1, numHidden, nonlinear, nonlinearDeriv)
    outputLayer = FullyConnectedLayer(numHidden, numOutput, nonlinear, nonlinearDeriv)

    # Until we're done,
    for i in xrange(numIterations):
        for samplei in xrange(len(samplesToProc)):
            sample = samplesToProc[samplei]
            # Do forward propagation on this sample.
            inputOut = inputLayer.forward_prop(sample)
            hiddenOut = hiddenLayer.forward_prop(inputOut)
            outOut = outputLayer.forward_prop(hiddenOut)

            # Now, do backward propagation.
            outDelta = np.subtract(desired[samplei], outOut)

            outBack = outputLayer.backward_prop(outDelta, learningRate, momentum)
            # The output from the last back prop layer isn't useful.
            hiddenLayer.backward_prop(outBack, learningRate, momentum)

            # Print the current error we're experiencing.
            if i % 100 == 0:
                print "error on iteration %d: %0.3f" % (i, Error(desired[samplei], outOut))

    # Print our final values.
    print "FINAL VALUES:"
    for i in xrange(len(samplesToProc)):
        inp = samplesToProc[i]
        print "input: ", inp

        # Get the final value.
        inputOut = inputLayer.forward_prop(inp)
        hiddenOut = hiddenLayer.forward_prop(inputOut)
        outOut = outputLayer.forward_prop(hiddenOut)

        # Print final along with expected, and error.
        print "output: ", outOut
        print "expected output: ", desired[i]
        print "error: %0.3f" % Error(desired[i], outOut)

if __name__ == "__main__":
    FullyConnectedLayerBatchTest()
