"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5 # To threshold the sigmoid 
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        sigomode=1/(1+np.exp(-1*z))
        return sigomode


    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = weights
        N, D = X_train.shape
        onehot=np.zeros([10,N], dtype=int)
        for x in range(N):
            label=y_train[x]
            onehot[label][x]=2
        onehot=onehot-1        
        for y in range(10):
            summ=0
            for z in range(N):
                summ=summ+self.sigmoid(-1*np.dot(np.dot(self.w[y][:],X_train[z][:]),onehot[y][z]))*np.dot(onehot[y][z],X_train[z][:])
            summ=summ/N
            self.w[y][:]=self.w[y][:]+self.lr*(self.weight_decay*self.w[y][:]+summ) 
        return self.w


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        rows, cols = X_test.shape
        prediction=np.zeros((rows,), dtype=int)
        
        for x in range(rows):
            #pred=0
            added=np.dot(self.w[0][:],X_test[x][:])
            for i in range(10):
                if (np.dot(self.w[i][:],X_test[x][:]))>added:
                    added=np.dot(self.w[i][:],X_test[x][:])
                    prediction[x]=i
        return prediction
