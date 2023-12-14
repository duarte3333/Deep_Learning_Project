#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        y_predict = np.dot(self.W, x_i) 
        predicted_label = np.argmax(y_predict)

        if predicted_label != y_i:
            learning_rate = kwargs.get('learning_rate', 0.001)
            self.W[y_i, :] += learning_rate * x_i
            self.W[predicted_label, :] -= learning_rate * x_i
            
#Worse than sklearn because it doesn't shuffle the data, so it's not stochastic.
#Why sklearn is so better? Use convergence acceleration methods and a better stopping criterion.


class LogisticRegression(LinearModel):

    def softmax(self, y_predict):
        return np.exp(y_predict) / np.sum(np.exp(y_predict), axis=0)

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Calculate the predicted probabilities using softmax
        y_predict = np.expand_dims(self.W.dot(x_i), axis = 1) #expand_dims(): explicitly represent the result as a column vector.
        
        # One-hot encode true label (num_labels x 1).
        y_i_one_hot = np.zeros((np.size(self.W, 0), 1)) #4x1
        y_i_one_hot[int(y_i)] = 1
        
        predicted_probs = self.softmax(y_predict)
        
        # Calculate the error
        error = predicted_probs - y_i_one_hot #ex: [0.25, 0.25, 0.25, 0.25] - [0, 0, 1, 0]
        
        # Update weights using stochastic gradient descent
        # How much the loss would change with respect to each weight.
        loss_gradient = np.dot(error, np.expand_dims(x_i, axis=1).T) #4x1 * 1x784 = 4x784 (4 classes by 784 features)
        self.W -= learning_rate * loss_gradient

        # Calculate and return the negative log-likelihood loss
        loss = -np.sum(y_i * np.log(predicted_probs)) #-SUM(log(softmax(y_predict))) ->cross entropy loss
        return loss
    
    # By using the expand_dims() - case 1:
    # Shape:
    #     Case 1 results in a 2D array with shape (num_classes, 1).
    #     Case 2 results in a 1D array with shape (num_classes,).

    # Dimensionality:
    #     Case 1 introduces an additional dimension, making it a column vector.
    #     Case 2 is a flat array, essentially a row vector.


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        #loc is the mean, scale is the standard deviation, and size is the number of elements you want to generate.
        #std = sqrt(0.01) = 0.1
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))
        self.b1 = np.zeros((hidden_size))
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        self.b2 = np.zeros((n_classes))
        self.weights = [self.W1, self.W2]
        self.biases = [self.b1, self.b2]
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, y_predict):
        return np.exp(y_predict) / np.sum(np.exp(y_predict))
    
    def predict(self, X):
        # Compute the forward pass of the network.
        h0 = X.T
        z1 = self.W1.dot(h0) + self.b1
        h1 = self.relu(z1)
        z2 = self.W2.dot(h1) + self.b2
        predicted_probs = self.softmax(z2)
        predicted_labels = predicted_probs.argmax(axis=0)
        return predicted_labels
    
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
    
    def forward(self, x, weights, biases):
        num_layers = len(weights) #2
        hiddens = []
        g = np.tanh
        # compute hidden layers
        #print("x_shape", x.shape, "weights_shape", weights[0].shape, "biases_shape", biases[0].shape)
        for i in range(num_layers):
                h = x if i == 0 else hiddens[i-1]
                #print("weights: ", weights[i].shape, "h: ", h.shape, "biases: ", biases[i].shape)
                z = weights[i].dot(h) + biases[i]
                if i < num_layers-1:  # Assuming the output layer has no activation.
                    hiddens.append(g(z))
                #print("z: ", z.shape)
        #compute output
        output = z
        return output, hiddens
    
    def compute_loss(self, output, y):
        # compute loss
        #print("y: ", y.shape)
        probs = np.exp(output) / np.sum(np.exp(output))
        #print("probs: ", probs.shape)
        loss = -y.dot(np.log(probs))
        
        return loss  
    
    def backward(self, x, y, output, hiddens, weights):
        num_layers = len(weights)
        #g = np.tanh
        z = output

        #probs = self.softmax(z)
        probs = np.exp(output) / np.sum(np.exp(output))
        #print("x_train: ", x.shape)
        #print("y_train: ", y.shape, "probs: ", probs.shape)
        grad_z = probs - y
        #print("grad_z: ", grad_z.shape)
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1): #(start, stop, step)
            
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)
            
            # Gradient of hidden layer below.
            grad_h = weights[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases
    
    def one_hot(self, y):
        one_hot = np.zeros((np.size(y, 0), self.W2.shape[0]))
        for i in range(np.size(y, 0)):
            one_hot[i, y[i]] = 1
        return one_hot
    
    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        #y = np.expand_dims(y, axis=1)
        y = self.one_hot(y)
        #print("X: ", X.shape, "y: ", y.shape, "W1: ", self.W1.shape, "W2: ", self.W2.shape)

        num_layers = len(self.weights)
        total_loss = 0
        # For each observation and target
        for x_train, y_train in zip(X, y): #x_train: 784x1, y_train: 4x1
            # Compute forward pass
            output, hiddens = self.forward(x_train, self.weights, self.biases) #output: 4x1
            
            # Compute Loss and Update total loss
            loss = self.compute_loss(output, y_train) # Compute Loss and Update total loss
            total_loss+=loss
            
            # Compute backpropagation
            grad_weights, grad_biases = self.backward(x_train, y_train, output, hiddens, self.weights)
            
            # Update weights
            for i in range(num_layers):
                self.weights[i] -= learning_rate*grad_weights[i]
                self.biases[i] -= learning_rate*grad_biases[i]
        return total_loss


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
