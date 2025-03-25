# MNIST Digit Recognition using Neural Networks

## Overview
This project implements feedforward neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The network is trained using stochastic gradient descent (SGD) and backpropagation.
![mnist_sample](https://github.com/user-attachments/assets/79c8c606-4d0b-4e2f-b8c0-5ac188eced8d)

## Features
- Implements a **fully connected neural network** from scratch using NumPy.
- Uses **sigmoid activation function**.
- Trains using **stochastic gradient descent (SGD)**.
- Implements **backpropagation** for weight and bias updates.
- Evaluates accuracy on test data.

## Requirements
Make sure you have the following dependencies installed:
```bash
pip install numpy
```

## Code Explanation
### Neural Network Structure
The neural network consists of multiple layers:
- **Input layer**: 784 neurons (corresponding to 28×28 grayscale pixel values).
- **Hidden layers**: Two layers with 30 and 25 neurons, respectively.
- **Output layer**: 10 neurons (representing digits 0-9).

The model parameters include:
- **Weights** (`self.weights`): Randomly initialized weight matrices for each layer.
- **Biases** (`self.biases`): Randomly initialized biases for each neuron.

### Feedforward
Each layer processes the input through:
1. Weighted sum:
   <img width="1045" alt="Screen Shot 2025-03-24 at 11 36 55 PM" src="https://github.com/user-attachments/assets/9bcc5cab-33f0-415a-8fd8-f2e4348077de" />

2. Activation function:
   <img width="1045" alt="Screen Shot 2025-03-24 at 11 38 35 PM" src="https://github.com/user-attachments/assets/aa7f4ac5-e444-4a27-b2cd-88b65f17e053" />


### Backpropagation
The backpropagation algorithm computes the gradients for updating weights and biases.

1. **Output layer error (BP1):**
   <img width="1045" alt="Screen Shot 2025-03-24 at 11 33 47 PM" src="https://github.com/user-attachments/assets/47c46e54-1c37-4568-98c6-024082ad33e0" />

   The first term on the right, ∂C/∂aLj, just measures how fast the cost is changing as a function of the jth
   output activation. If, for example, C
   doesn't depend much on a particular output neuron, j
  , then δLj
   will be small, which is what we'd expect. The second term on the right, σ′(zLj)
   measures how fast the activation function σ
   is changing at zLj.

   where:
   - \( C \) is the cost function.
   - \( sigma_prime(z) \) is the derivative of the sigmoid function.
   - \( dot multip \) denotes element-wise multiplication.

2. **Gradient for the output layer weights (BP4):**
   <img width="1045" alt="Screen Shot 2025-03-24 at 11 31 46 PM" src="https://github.com/user-attachments/assets/3a986878-5301-464e-ba5e-48077d9bf08c" />

   This tells us how to compute the partial derivatives ∂C/∂wljk
   in terms of the quantities δl
   and al−1
  , which we already know how to compute.

3. **Backpropagating error through layers (BP2 & BP3):**
     <img width="1045" alt="Screen Shot 2025-03-24 at 11 33 35 PM" src="https://github.com/user-attachments/assets/73e57396-2891-42a6-ae2a-5e5640fa58dc" />

     where (wl+1)T
     is the transpose of the weight matrix wl+1
     for the (l+1)th
     layer. This equation appears complicated, but each element has a nice interpretation. Suppose we know the error δl+1
     at the l+1th
     layer. When we apply the transpose weight matrix, (wl+1)T
    <img width="1045" alt="Screen Shot 2025-03-24 at 11 33 11 PM" src="https://github.com/user-attachments/assets/affec91b-4a88-4f9c-9aa2-c3038a510d84" />

    That is, the error δlj
    is exactly equal to the rate of change ∂C/∂blj.
    This is great news, since (BP1) and (BP2) have already told us how to compute δlj.

   - This propagates errors backward through the network.

3. **Updating weights and biases:**
    <img width="1045" alt="Screen Shot 2025-03-24 at 11 29 28 PM" src="https://github.com/user-attachments/assets/38865af6-4e12-4376-b340-af6cdff93390" />

     where :
     - \( eta \) is the learning rate.

### Training with Stochastic Gradient Descent (SGD)
The dataset is divided into mini-batches, and the network is updated using:
1. Shuffle training data.
2. Process mini-batches and update weights/biases.
3. Evaluate accuracy on validation data.

### Running the Model
The network is trained as follows:
```python
training_data, test_data, valid_data = load_data.loader()
obj = Network([784, 30, 25, 10])
obj.SGD(training_data, 50, 10, 1.3, valid_data)
```
- **50 epochs**: Number of times the training data is processed.
- **Batch size = 10**: Training is performed on mini-batches of size 10.
- **Learning rate = 1.3**: Determines step size during weight updates.

### The Best Result of the model is about 95.24%
During training, the model prints the accuracy after each epoch:
```
(env) bel-oirg@e2r9p12 neural_network % python3 neural.py
Epoch 0 : 8923 / 10000
Epoch 1 : 9012 / 10000
...
Epoch 45 : 9519 / 10000
Epoch 46 : 9524 / 10000
Epoch 47 : 9514 / 10000
```
This indicates how many digits were correctly classified.

## Conclusion
This project demonstrates how a simple neural network can classify handwritten digits using backpropagation and stochastic gradient descent. It serves as a foundation for understanding more advanced deep learning architectures.

## References
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) (included in the repo)

