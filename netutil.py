import math
import json
import network as netlib

def sigmoid(number):
    return 1/(1+math.exp(-number))
pass

def logit(number):
    return math.log(number/(1-number))
pass

def logit_arr(array):
    for i in range(array):
        array[i] = logit(array[i])
    pass
    return array
pass

def load(path):
    file = open(path, "r")
    data = file.read()
    file.close()
    jsonO = json.loads(data)
    network = netlib.Network(jsonO['input_neuron_count'], jsonO['hidden_neuron_count'], jsonO['output_neuron_count'])
    for i in range(len(jsonO['weights'])):
        layer = jsonO['weights'][i]
        for j in range(len(layer)):
            neuron = layer[j]
            for k in range(len(neuron)):
                weight = neuron[k]
                network.layers[i].neurons[j].weights[k] = weight
            pass
        pass
    pass
    return network
pass

def save(network, path):
    i = []
    for l in range(len(network.layers)):
        if l == 0:
            continue
        layer = network.layers[l]
        j = []
        for neuron in layer.neurons:
            k = []
            for weight in neuron.weights:
                k.append(weight)
            pass
            j.append(k)
        pass
        i.append(j)
    pass
    hn = []
    for layer in range(1, len(network.layers) - 1):
        hn.append(len(network.layers[layer].neurons))
    out = {
        'input_neuron_count': len(network.layers[0].neurons),
        'hidden_neuron_count': hn,
        'output_neuron_count': len(network.layers[len(network.layers)-1].neurons),
        'weights': i
    }
    jsonO = json.dumps(out, indent=4, sort_keys=True)
    file = open(path, "w")
    file.write(jsonO)
    file.close()
pass
