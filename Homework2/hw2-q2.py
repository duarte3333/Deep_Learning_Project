#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    #image size: 28x28
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        num_classes = 4
        # Convolutional layers
        if not no_maxpool:
            # Implementation for Q2.1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # The size of the image after two convolutions and two max pooling will be:
            # (28 - 3 + 2*1) / 1 + 1 = 28 , conv1
            # (28 - 2) / 2 + 1 = 14       , max pool1
            # (14 - 3 + 2*0)/1 + 1 = 12   , conv2
            # (12 - 2)/2 + 1 = 6          , max pool2
            self.fc1_input_size = 16 * 6 * 6 #16 channels, 6x6 image
        else:
            # Implementation for Q2.2
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
            # The size of the image after two convolutions with stride of 2 will be:
            # (28 + 3 - 2*1)/2 + 1 = 14  , conv1
            # (14 - 3) / 2 + 1 = 6       , conv2
            self.fc1_input_size = 16 * 6 * 6
        
        # Fully connected layers
        # Implementation for Q2.1 and Q2.2
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=320)
        self.fc2 = nn.Linear(in_features=320, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)

        # Dropout layer
        self.drop = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # input should be of shape [b, c, w, h]
        # Check if input needs to be reshaped from [b, 784] to [b, c, w, h]
        #if x.dim() == 2 and x.shape[1] == 784:
        # Reshape input to [b, c, w, h]
        x = x.view([-1, 1, 28, 28])
        #print("x_shape:", x.shape)
        #print("starting shape - x_shape:", x.shape)
        # conv and relu layers
        x = F.relu(self.conv1(x))
        #print("after conv1: ", x.shape)
        
        if not self.no_maxpool:
            # max-pool layer if using it
            x = self.pool(x)
            #print("max-pool1: ", x.shape)
            
        # conv and relu layers
        x = F.relu(self.conv2(x))
        #print("after conv2: ", x.shape)
        if not self.no_maxpool:
            # max-pool layer if using it
            x = self.pool(x)
            
        #print("after max-pool2", x.shape)
        
        # prep for fully connected layer + relu
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.fc1_input_size)
        
        # Fully connected layer 1 + relu
        x = F.relu(self.fc1(x))

        # drop out
        x = self.drop(x)

        # Fully connected layer 2 + relu
        x = F.relu(self.fc2(x))

        # last fully connected layer
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    
    ##train your model for 15 epochs using SGD tuning only the learning rate on your validation
    ##data, using the following values: 0.1, 0.01, 0.001.

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=15, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
