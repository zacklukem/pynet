import cmath
from random import uniform

class Network:
    # Network constructor
    # @param inputNeurons [int] number of input neurons
    # @param hiddenNeurons [int[]] number of hidden neurons and layers
    # @param outputNeurons [int] number of output neurons
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons):
        self.layers = []
        self.layers.append(Layer(inputNeurons))
        for i in range(len(hiddenNeurons)):
            self.layers.append(Layer(hiddenNeurons[i]))
        self.layers.append(Layer(outputNeurons))

    def calc(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].neurons[i].value = inputs[i]

        for i in range(1, len(self.layers)):
            self.layers[i].compute(self.layers[i - 1])

        return self.layers[len(self.layers) - 1]

    def propagate(self, output, learningRate):
        outputLayer = self.layers[len(self.layers) - 1]
        inputLayer = self.layers[0]

        for l in range(1, len(self.layers) - 1):
            hiddenLayer = self.layers[l]

            i = 0.0
            hiddenNeuron = Neuron(0)
            value = 0.0
            sum = 0.0
            gradient = 0.0
            for i in range(0, len(output)):
                hiddenNeuron = outputLayer.neurons[i]
                value = outputLayer.neurons[i].value
                w = output[i] - value

                for inputValue in range(len(hiddenNeuron.weights)):
                    hiddenNeuron1 = hiddenLayer.neurons[inputValue]
                    sum = hiddenNeuron1.value
                    gradient = -value * (1.0 - value) * sum * w
                    hiddenNeuron.weights[inputValue] -= learningRate * gradient

            for i in range(len(hiddenLayer.neurons)):
                hiddenNeuron = hiddenLayer.neurons[i]
                value = hiddenNeuron.value

                for var28 in range(len(hiddenNeuron.weights)):
                    inputNeuron = inputLayer.neurons[var28]
                    var29 = inputNeuron.value
                    sum = 0.0

                    for var30 in range(len(outputLayer.neurons)):
                        outputNeuron = outputLayer.neurons[var30]
                        outputWeight = outputNeuron.weights[i]
                        outputValue = outputNeuron.value
                        outputError = outputValue - output[var30]
                        outputGradient = outputError * outputValue * (1.0 - outputValue) * outputWeight
                        sum += outputGradient

                    gradient = value * (1.0 - value) * var29 * sum
                    hiddenNeuron.weights[var28] -= learningRate * gradient

    # Network toString
    def __str__(self):
        o = "Network of size " + str(len(self.layers)) + ":"
        for layer in self.layers:
            o += "\n\t" + str(layer)
        return o

class Neuron:
    def __init__(self, weights):
        self.value = 0.0
        self.weights = []
        for _ in range(weights):
            self.weights.append(uniform(-1, 1))

    def __str__(self):
        return "<" + str(self.value) + ">";

class Layer:
    def __init__(self, size):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron(size))

    def compute(self, layer):
        for i in range(len(self.neurons)):
            sum = 0.0
            for j in range(len(layer.neurons)):
                sum += layer.neurons[j].value * self.neurons[i].weights[j]

            self.neurons[i].value = 1/(1 + cmath.exp(-sum))

    def __str__(self):
        o = "Layer of size " + str(len(self.neurons)) + ":"
        for neuron in self.neurons:
            o += "\n\t" + str(neuron)
        return o
