from .base_layer import BaseLayer
import numpy as np

class CrossEntropyLoss(BaseLayer):
    def __init__(self):
        self.cache = None
        pass

    def forward(self, input_x: np.ndarray, target_y: np.ndarray
    ):
        """
        TODO: Implement the forward pass for cross entropy loss function

        """
        N, _ = input_x.shape
        # Calculate the sum of losses for each example, loss for one example -log(e_i/sum(e_j)) where i is the 
        # correct class according to the label target_y and j is sum over all classes
        loss=np.sum(-np.log(input_x[np.arange(len(input_x)), target_y]))
        #for x in range(len(target_y)):
            #loss=loss-np.log(input_x[x][target_y[x]])
        # Normalize the loss by dividing by the total number of samples N
        loss /= N
        #Store your loss output and input and targets in cache
        self.cache = [loss.copy(), input_x.copy(), target_y.copy()]
        return loss

    def backward(self):
        """
        TODO: Compute gradients given the true labels
        """
        # Retrieve data from cache to calculate gradients
        loss_temp, x_temp, y_temp = self.cache
        N, _ = x_temp.shape
        y=np.zeros(x_temp.shape, dtype=int)
        #for x in range(len(y_temp)):
            #y[x][y_temp[x]]=1
        y[np.arange(len(y)), y_temp] =1
        #y[:len(y_temp)][y_temp]=1
        # Use the formula for the gradient of Cross entropy loss to calculate the gradients
        # Gradient matrix will be NxD matrix, with N rows for all the samples in the minibatch, and D=3072
        dx = x_temp-y
        assert x_temp.shape==dx.shape, "Mismatch in shape"
        # Normalize the gradient by dividing with the total number of samples N
        dx /= N
        return dx

    def zero_grad(self):
        pass
