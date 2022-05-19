from .base_layer import BaseLayer
import numpy as np


class Linear(BaseLayer):
    '''Linear Neural Network layers'''
    def __init__(
        self,
        input_dims: int,
        output_dims: int
    ):
        # Initialize parameters for the linear layer randomly
        self.w = np.random.rand(input_dims, output_dims) * 0.0001
        self.b = np.random.rand(output_dims) * 0.0001
        self.dw = None
        self.db = None
        self.cache = None

    def forward(
        self,
        input_x: np.ndarray
    ):
        # TODO: Implement forward pass through a single linear layer, similar to the linear regression output
        # Output = dot product between W and X and then add the bias
        #print(self.w.shape)
        #print(input_x.shape)
        #print(self.b.shape)
        #z=len(self.w[0])
        #y=len(self.w)self
        output=np.dot(input_x,self.w)+self.b
        #for x in range(z):
            #output[0][z] = np.sum(np.dot(self.w[:][z],input_x))+self.b[z]
            #print(input_x.shape)
        # Store the arrays in cache, useful for calculating the gradients in the backward pass
        #print(input_x.shape)
        self.cache = [input_x.copy(), self.w.copy(), self.b.copy()]
        return output

    def backward(self, dout):
        # TODO: Implement backward pass to calculate gradients for W and X, that is dw and dx
        # dw and dx can be estimated from the incoming gradient dout, using chain rule as discussed in class
        temp_x, temp_w, _ = self.cache
        dx=np.dot(dout, temp_w.T)
        self.dw=np.dot(temp_x.T,dout)
        self.db=np.sum(dout,axis=0)
        return dx

    def zero_grad(self):
        # Reinitialize the gradients
        self.dw = None
        self.db = None

    @property
    def parameters(self):
        return [
            self.w,
            self.b
        ]

    @property
    def grads(self):
        return [
            self.dw,
            self.db
        ]