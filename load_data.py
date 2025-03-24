import numpy as np 
import pickle
import gzip

def vectorize(i):
    v = np.zeros((10, 1))
    v[i] = 1.0
    return v
    
# print(train_x[0].shape)
#it returns (784,) cause, NumPy drops the second dimension when slicing a single row by default.

def loader():
    f = gzip.open('/Users/bel-oirg/Desktop/neural_network/mnist.pkl.gz', 'rb')
    train, test, valid = pickle.load(f, encoding='latin1')

    train_x, train_y = train
    test_x, test_y = test
    valid_x, valid_y = valid

    training_inp = [np.reshape(x, (784, 1)) for x in train_x]
    training_res = [vectorize(i) for i in train_y]
    training_data = list(zip(training_inp, training_res))
    #50k elements

    test_inp = [np.reshape(x, (784, 1)) for x in test_x]
    test_data = list(zip(test_inp, test_y))
    #10k elements

    valid_inp = [np.reshape(x, (784, 1)) for x in valid_x]
    valid_data = list(zip(valid_inp, valid_y))
    #10k emements

    return training_data, test_data, valid_data