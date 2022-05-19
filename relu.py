from .base_layer import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        self.cache = None

    def forward(self, input_x: np.ndarray):
        # TODO: Implement RELU activation function forward pass
        input_x[input_x<0]=0
        output=input_x
        self.cache = input_x.copy()
        return output

    def backward(self, dout):
        # Load the input from the cache
        x_temp = self.cache
        # Calculate gradient for RELU 
        x_tempo=(x_temp>0).astype(np.uint8)
        dx=x_tempo*dout
        return dx
