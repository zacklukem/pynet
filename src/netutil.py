import cmath

def sigmoid(number):
    return 1/(1+cmath.exp(-number))

def logit(number):
    return cmath.log(number/(1-number))
