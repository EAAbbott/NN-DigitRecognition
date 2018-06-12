
import random
import numpy as np


# Define activation function f() and f'() 
def sigmoid(x):
     return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x)) 

class N_network:
    """
    Neural Network (NN).
    Size - Network size [In, ..nLayers.., Out] (at least length 3 list)
    """
    def __init__(self, size):
        self.size = size
        self.n_layers = len(size)
        self.weights = []
        self.biases = []
        
    def init_weights_rand(self, seed=None):
        """
        Initialise weights randomly via numpy.random.randn with default 
        parameters (mean about 0 with std.dev of 1).
        Weight matrix shape is defined as backwards connections to remove the 
        need to transpose during backpropagation.
        As biases are a node output bias, they are not needed on the 1st 
        layer (input layer)
        """
        np.random.seed(seed)
        self.weights = [np.random.randn(n, m) 
                        for n, m in zip(self.size[1:], self.size[:-1])]
        self.biases = [np.random.randn(n, 1)
                        for n in self.size[1:]]
        
    def train_network(self, training_data, batch_size,
                      cycles, eta, test_data=None):
        """
        training data - list of tupled image and label data.
        cycles - number of times all training data is run through (AKA epochs).
        batch_size - size of batch after which weight/bias updates are applied.
        eta - learning rate/ step size
        test data - whether to check accuracy against test data.
        """
        n_data = len(training_data)
        n_test = len(test_data) if test_data else None
        # for each cycle/epoch
        for i in range(cycles):
            # randomly shuffle data for current cycle.
            random.shuffle(training_data)
            # split training data into batches of batch_size
            batches = []
            for j in range(0, n_data, batch_size):
                batches.append(training_data[j:(j + batch_size)])
            #for each batch
            for batch in batches:
                # update network weights and biases for current batch
                self.update_batch(batch, eta)
            if test_data:
                no_correct = self.test_network(test_data)         
                print("Cycle {0:>2}: {1:,} / {2:,} = {3:.2%} accuracy".format(
                        i, 
                        no_correct, 
                        n_test, 
                        (no_correct / n_test)))
            else:
                print("Cycle {0} complete".format(i))
                
    def test_network(self, test_data):
        """
        Tests input data and returns number of correct outputs.
        (Generic testing function)
        """
        results = [(np.argmax(self.feed_forward(x)), y)
                    for x, y in test_data]
        return sum(int(x == y) for x, y in results)
                
    def backpropagation(self, x, y):
        """
        x training data, y training label.
        Returns tuplpe (del_b, del_w), gradients of the 
        cost function for b and w.
        Output shapes correspond to self.biases and self.weights.
        """
        # Initialise gradients to 0.
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        # calculate the z vector and activation vector for each layer.
        activation = x
        activations = [x] # list to store activation vectors for each layer.
        # list to store z input vectors for each layer
        z_list = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # delta used in w and b gradients
        delta = 2. * \
                (activations[-1] -  y) * \
                sigmoid_der(z_list[-1])
        # b gradient = delta
        del_b[-1] = delta
        # w gradient = delta * (a^(L-1)) (previous layer's activations)
        del_w[-1] = np.dot(delta, activations[-2].T)
        # back propagate delta values using L+1 variant
        for i in range(2, self.n_layers):
            # take z from previous layer
            z = z_list[-i]
            # calculate sigmoid derivative of z
            sig_prime_z = sigmoid_der(z)
            # calculate delta 
            delta = np.dot(self.weights[-i + 1].T, delta) * sig_prime_z
            del_b[-i] = delta
            del_w[-i] = np.dot(delta, activations[-i - 1].T)
        return(del_b, del_w)

    def update_batch(self, batch, eta):
        """
        Applies changes to the networks weights and biases using deltas from
        backpropagation method.
        """
        # initialise change matrices to 0
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        # for each image in batch, backpropagate delta, then update gradient
        for x, y in batch:
            # gather deltas for network
            delta_del_b, delta_del_w = self.backpropagation(x, y)
            # update b and w gradients for the batch
            del_b = [grad_bias + delta_grad_bias 
                     for grad_bias, delta_grad_bias in zip(del_b, delta_del_b)]
            del_w = [grad_weight + delta_grad_weight 
                     for grad_weight, delta_grad_weight in zip(del_w,
                                                               delta_del_w)]
        # Update weights and biases.
        self.weights = [w - (eta / len(batch)) * grad_w
                        for w, grad_w in zip(self.weights, del_w)]
        self.biases = [b - (eta / len(batch)) * grad_b
                       for b, grad_b in zip(self.biases, del_b)]
        
        
    def feed_forward(self, input_data=None):
        """
        Input data multiplied through each layer of the networks weights plus 
        the biases. 
        Output is last layer(network output) activation values,
        """
        a = input_data
        # For each layer (minus input), dot product of input and weights gives
        # 1 x n array of activations (less the bias) for that layer.
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a