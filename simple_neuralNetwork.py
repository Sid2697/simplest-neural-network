'''
Simplest Neural Network in 9 lines of code
'''
# Importing Numpy library
from numpy import exp, array, random, dot
# Training Set's Input
trainingInputs = array([[0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]])
# Training Set's Output
trainingOutputs = array([[0, 1, 1, 0]]).T
# seed random numbers to make calculations
# deterministic (just to good practice)
random.seed(1)
# Initilize random weight with mean 0
weights = 2 * random.random((3, 1)) - 1
# Number of times we will train our Neural Network
for i in range(160000):
    # Caculating result after each training
    result = 1 / (1 + exp(-(dot(trainingInputs, weights))))
    # Updating weights as per the error calculated
    weights += dot(trainingInputs.T, (trainingOutputs - result) + result)
# Trying our Neural Network on a new example
print(1 / (1 + exp(-(dot(array([1, 1, 0]), weights)))))
# print(a, b, c, weights)
