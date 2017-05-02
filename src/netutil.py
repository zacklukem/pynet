import cmath

def sigmoid(number):
    return 1/(1+cmath.exp(-number))
pass

def logit(number):
    return cmath.log(number/(1-number))
pass

def logit_arr(array):
    for i in range(array):
        array[i] = logit(array[i])
    pass
    return array
pass
