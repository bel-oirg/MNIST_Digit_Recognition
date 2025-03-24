import random
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
import numpy as np 

import load_data

def sigmoid(x):#
    return 1.0 / (1.0 + np.exp(-x))
    #used np.exp() and not the regular exp on math lib,
    #be cause most of the time x is a vector or array

def sigmoid_prime(x):#
    return (sigmoid(x) * (1 - sigmoid(x)))

class Network():
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x, 1)
                        for x in sizes[1:]]

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate_result(self, test_data):
        vec = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in test_data]

        # for (x, y) in test_data:
        #     print(f'||{self.feedforward(x)}"""')

        # print(str(int(x) for x in vec))
        return sum(int(x == y)  for (x, y) in vec)


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(x.shape) for x in self.biases]
        nabla_w = [np.zeros(x.shape) for x in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for dnb, nb in zip(delta_nabla_b, nabla_b)]
            nabla_w = [nw + dnw for dnw, nw in zip(delta_nabla_w, nabla_w)]
        
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for nw, w in zip(nabla_w, self.weights)]
    
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for nb, b in zip(nabla_b, self.biases)]



    def cost_deriv(self, a, y):
        return (a - y)
    
    def backprop(self, x, y):
        activation = x
        activations = [x]
        zs = []

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #BP1
        delta = self.cost_deriv(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta

        #BP4
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            
            #BP3
            nabla_b[-l] = delta
            
            #BP4
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w


    def SGD(self, train_data, epoch, batch_size, eta, test_data=None):
        train_len = len(train_data)
        if test_data:
            test_len = len(test_data)
        for i in range(epoch):
            random.shuffle(train_data)

            batched = [train_data[k:k+batch_size]
                       for k in range(0, train_len, batch_size)]

            for mini_batch in batched:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f'Epoch {i} : {self.evaluate_result(test_data)} / {test_len}')
            else:
                print(f'Epoch {i} is done.')


training_data, test_data, valid_data = load_data.loader()
obj = Network([784, 30, 25, 10])

obj.SGD(training_data, 50, 10, 1.3, valid_data)