#!/usr/bin/env python3
from network import *
from random import uniform

network = Network(2, [2], 2) # new network with 2 inputs 1 layer of 2 hiddens and 2 outputs

for i in range(10000):
    network.calc([uniform(-1, 1), uniform(-1, 1)]) # Train with random numbers
    network.propagate([0.5, 0.8], 0.1) # expected output, learningRate

network.calc([uniform(-1, 1), uniform(-1, 1)]) # test with random number

print(network) # print values