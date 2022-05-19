"""
Linear Regression model
"""

import numpy as np

class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None #Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay


    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        
        self.w = weights
        N, D = X_train.shape
        onehot=np.zeros([10,N], dtype=int)
        for x in range(N):
            label=y_train[x]
            onehot[label][x]=1       
        for y in range(10):          
            summ=0
            for z in range(N):
                summ=summ+2*np.dot((np.dot(self.w[y][:],X_train[z][:])-onehot[y][z]),X_train[z][:])
            summ=summ/N
            self.w[y][:]=self.w[y][:]-self.lr*(self.weight_decay*self.w[y][:]+summ) 
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
            added=np.dot(self.w[0][0:3071],X_test[x][0:3071])+self.w[0][3072]
            for i in range(10):
                if (np.dot(self.w[i][0:3071],X_test[x][0:3071])+self.w[0][3072])>added:
                    added=np.dot(self.w[i][0:3071],X_test[x][0:3071])+self.w[0][3072]
                    prediction[x]=i
        return prediction
        