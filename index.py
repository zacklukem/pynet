#!/usr/bin/env python3
from random import uniform
from network import *
import netutil as util



network = util.load("i_like_0_5.json") # hiddens and 2 outputs

for i in range(10000):
    network.calc([                      # Train with random numbers
        uniform(-1, 1),
        uniform(-1, 1)
    ])
    network.propagate([0.5, 0.8], 0.1)  # expected output, learningRate

network.calc([
    uniform(-1, 1),
    uniform(-1, 1)
])                                      # test with random number

print(network)                          # print values

util.save(network, "i_like_0_5.json")
