"""
Core neural network layer implementations for building custom neural networks from scratch.
This file contains implementations of various layer types that can be used to construct
neural network architectures, with each layer having forward and backward pass implementations.
"""

import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    
    Each layer must implement a forward pass for inference and a backward pass
    for gradient computation during backpropagation.
    """
    def __init__(self):
        self.params = {}  # Dictionary to store trainable parameters
        self.grads = {}   # Dictionary to store gradients of trainable parameters
        
    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass computation.
        
        Args:
            inputs: Input data to the layer
            
        Returns:
            The output of the layer
        """
        pass
    
    @abstractmethod
    def backward(self, dout):
        """
        Backward pass computation.
        
        Args:
            dout: Gradient of the loss with respect to the layer's output
            
        Returns:
            Gradient of the loss with respect to the layer's input
        """
        pass

class Dense(Layer):
    """
    Fully connected layer implementation.
    
    This layer implements the operation: outputs = inputs * weights + biases
    """
    def __init__(self, input_size, output_size):
        """
        Initialize a fully connected layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        super().__init__()
        # Initialize weights with small random values to break symmetry
        self.params['W'] = np.random.randn(input_size, output_size) * 0.01
        # Initialize biases with zeros
        self.params['b'] = np.zeros(output_size)
        
    def forward(self, inputs):
        """
        Forward pass for the fully connected layer.
        
        Args:
            inputs: Inputs of shape (batch_size, input_size)
            
        Returns:
            Outputs of shape (batch_size, output_size)
        """
        # Store inputs for use in the backward pass
        self.inputs = inputs
        # Compute the linear transformation
        return np.dot(inputs, self.params['W']) + self.params['b']
    
    def backward(self, dout):
        """
        Backward pass for the fully connected layer.
        
        Args:
            dout: Gradient of loss with respect to the output, shape (batch_size, output_size)
            
        Returns:
            Gradient of loss with respect to the input, shape (batch_size, input_size)
        """
        # Compute gradient with respect to weights
        self.grads['W'] = np.dot(self.inputs.T, dout)
        # Compute gradient with respect to biases (sum across batch dimension)
        self.grads['b'] = np.sum(dout, axis=0)
        # Compute gradient with respect to inputs
        return np.dot(dout, self.params['W'].T)

class Conv2D(Layer):
    """
    2D Convolutional layer implementation from scratch.
    
    This layer applies a 2D convolution operation over an input signal composed of several input channels.
    """
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=0):
        """
        Initialize a 2D convolutional layer.
        
        Args:
            input_channels: Number of channels in the input image
            output_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters/kernels with small random values
        # Shape: (output_channels, input_channels, kernel_size, kernel_size)
        self.params['W'] = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size) * 0.01
        # Initialize biases with zeros
        self.params['b'] = np.zeros(output_channels)
        
    def forward(self, inputs):
        """
        Forward pass for the convolutional layer.
        
        Args:
            inputs: Inputs of shape (batch_size, input_channels, height, width)
            
        Returns:
            Outputs of shape (batch_size, output_channels, output_height, output_width)
        """
        # Store inputs for use in the backward pass
        self.inputs = inputs
        batch_size, channels, height, width = inputs.shape
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # Initialize output array
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))
        
        # Add padding if necessary
        if self.padding > 0:
            padded_inputs = np.pad(
                inputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                'constant'
            )
        else:
            padded_inputs = inputs
        
        # Perform convolution
        # Note: This is a naive implementation for clarity. 
        # In practice, you'd use im2col for efficiency.
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for h_out in range(out_height):
                    h_in = h_out * self.stride
                    for w_out in range(out_width):
                        w_in = w_out * self.stride
                        
                        # Extract the current patch
                        patch = padded_inputs[
                            b, 
                            :, 
                            h_in:h_in+self.kernel_size, 
                            w_in:w_in+self.kernel_size
                        ]
                        
                        # Calculate convolution (element-wise multiplication and sum)
                        output[b, c_out, h_out, w_out] = np.sum(
                            patch * self.params['W'][c_out]
                        ) + self.params['b'][c_out]
        
        return output
    
    def backward(self, dout):
        """
        Backward pass for the convolutional layer.
        
        Args:
            dout: Gradient of loss with respect to output, 
                  shape (batch_size, output_channels, output_height, output_width)
            
        Returns:
            Gradient of loss with respect to input,
            shape (batch_size, input_channels, input_height, input_width)
        """
        batch_size, _, out_height, out_width = dout.shape
        _, channels, height, width = self.inputs.shape
        
        # Initialize gradients
        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dinputs = np.zeros_like(self.inputs)
        
        # Add padding if necessary
        if self.padding > 0:
            padded_inputs = np.pad(
                self.inputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                'constant'
            )
            padded_dinputs = np.pad(
                dinputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                'constant'
            )
        else:
            padded_inputs = self.inputs
            padded_dinputs = dinputs
        
        # Calculate gradients
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for h_out in range(out_height):
                    h_in = h_out * self.stride
                    for w_out in range(out_width):
                        w_in = w_out * self.stride
                        
                        # Update dW (gradient of loss w.r.t. weights)
                        patch = padded_inputs[
                            b, 
                            :, 
                            h_in:h_in+self.kernel_size, 
                            w_in:w_in+self.kernel_size
                        ]
                        dW[c_out] += patch * dout[b, c_out, h_out, w_out]
                        
                        # Update db (gradient of loss w.r.t. biases)
                        db[c_out] += dout[b, c_out, h_out, w_out]
                        
                        # Update dinputs (gradient of loss w.r.t. inputs)
                        padded_dinputs[
                            b, 
                            :, 
                            h_in:h_in+self.kernel_size, 
                            w_in:w_in+self.kernel_size
                        ] += self.params['W'][c_out] * dout[b, c_out, h_out, w_out]
        
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # Remove padding if necessary
        if self.padding > 0:
            dinputs = padded_dinputs[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dinputs = padded_dinputs
        
        return dinputs

class MaxPool2D(Layer):
    """
    2D Max pooling layer implementation.
    
    This layer reduces the spatial dimensions of the input by taking the maximum value within
    sliding windows, helping with translation invariance and reducing computation.
    """
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize a max pooling layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Stride of the pooling operation
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, inputs):
        """
        Forward pass for the max pooling layer.
        
        Args:
            inputs: Inputs of shape (batch_size, channels, height, width)
            
        Returns:
            Outputs of shape (batch_size, channels, output_height, output_width)
        """
        self.inputs = inputs
        batch_size, channels, height, width = inputs.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output array
        output = np.zeros((batch_size, channels, out_height, out_width))
        # Store the indices of the maximum values for use in the backward pass
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    h_in = h_out * self.stride
                    for w_out in range(out_width):
                        w_in = w_out * self.stride
                        
                        # Extract the current patch
                        patch = inputs[
                            b, 
                            c, 
                            h_in:h_in+self.pool_size, 
                            w_in:w_in+self.pool_size
                        ]
                        
                        # Find max value and its indices within the patch
                        output[b, c, h_out, w_out] = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, h_out, w_out] = max_idx
        
        return output
    
    def backward(self, dout):
        """
        Backward pass for the max pooling layer.
        
        Args:
            dout: Gradient of loss with respect to output,
                  shape (batch_size, channels, output_height, output_width)
            
        Returns:
            Gradient of loss with respect to input,
            shape (batch_size, channels, input_height, input_width)
        """
        batch_size, channels, out_height, out_width = dout.shape
        _, _, height, width = self.inputs.shape
        
        # Initialize gradients
        dinputs = np.zeros_like(self.inputs)
        
        # Distribute gradients
        # For max pooling, gradient flows only through the maximum value in each window
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    h_in = h_out * self.stride
                    for w_out in range(out_width):
                        w_in = w_out * self.stride
                        
                        # Get max index within the patch
                        h_max, w_max = self.max_indices[b, c, h_out, w_out]
                        
                        # Set gradient only at the position that had the maximum value
                        dinputs[b, c, h_in+h_max, w_in+w_max] = dout[b, c, h_out, w_out]
        
        return dinputs

class Flatten(Layer):
    """
    Flatten layer implementation.
    
    This layer flattens the input, useful for transitioning from convolutional layers
    to fully connected layers.
    """
    def forward(self, inputs):
        """
        Forward pass for the flatten layer.
        
        Args:
            inputs: Inputs of any shape
            
        Returns:
            Flattened outputs of shape (batch_size, features)
        """
        # Store input shape for use in the backward pass
        self.input_shape = inputs.shape
        # Flatten all dimensions except the batch dimension
        return inputs.reshape(inputs.shape[0], -1)
    
    def backward(self, dout):
        """
        Backward pass for the flatten layer.
        
        Args:
            dout: Gradient of loss with respect to output, shape (batch_size, features)
            
        Returns:
            Gradient of loss with respect to input, restored to original shape
        """
        # Reshape gradient back to the input shape
        return dout.reshape(self.input_shape)

class Dropout(Layer):
    """
    Dropout layer implementation for regularization.
    
    This layer randomly sets a fraction of the input units to 0 during training,
    which helps prevent overfitting.
    """
    def __init__(self, drop_rate=0.5):
        """
        Initialize a dropout layer.
        
        Args:
            drop_rate: Fraction of the input units to drop (set to 0)
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward(self, inputs, training=True):
        """
        Forward pass for the dropout layer.
        
        Args:
            inputs: Inputs of any shape
            training: Boolean indicating whether in training mode
            
        Returns:
            Outputs with some values set to 0 (if training=True)
        """
        if training:
            # Generate binary mask with 1s for kept units and 0s for dropped units
            self.mask = np.random.binomial(1, 1-self.drop_rate, size=inputs.shape) / (1 - self.drop_rate)
            # Scale outputs by 1/(1-drop_rate) to maintain the expected sum
            return inputs * self.mask
        else:
            # During inference, no units are dropped
            return inputs
    
    def backward(self, dout):
        """
        Backward pass for the dropout layer.
        
        Args:
            dout: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # Apply the same mask to the gradient
        return dout * self.mask

class BatchNormalization(Layer):
    """
    Batch Normalization layer implementation.
    
    This layer normalizes the activations of the previous layer for each batch,
    which helps with training stability and convergence.
    """
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        """
        Initialize a batch normalization layer.
        
        Args:
            input_size: Number of features/channels
            momentum: Momentum for the running mean and variance
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Trainable parameters
        self.params['gamma'] = np.ones(input_size)  # Scale
        self.params['beta'] = np.zeros(input_size)  # Shift
        
        # Running estimates for inference
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
    
    def forward(self, inputs, training=True):
        """
        Forward pass for the batch normalization layer.
        
        Args:
            inputs: Inputs of shape (batch_size, features)
            training: Boolean indicating whether in training mode
            
        Returns:
            Normalized outputs of shape (batch_size, features)
        """
        self.inputs = inputs
        
        if training:
            # Calculate mean and variance over the batch dimension
            self.batch_mean = np.mean(inputs, axis=0)
            self.batch_var = np.var(inputs, axis=0)
            
            # Normalize
            self.x_norm = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Update running statistics for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running statistics for inference
            self.x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        return self.params['gamma'] * self.x_norm + self.params['beta']
    
    def backward(self, dout):
        """
        Backward pass for the batch normalization layer.
        
        Args:
            dout: Gradient of loss with respect to output, shape (batch_size, features)
            
        Returns:
            Gradient of loss with respect to input, shape (batch_size, features)
        """
        batch_size = self.inputs.shape[0]
        
        # Gradients for gamma and beta
        self.grads['gamma'] = np.sum(dout * self.x_norm, axis=0)
        self.grads['beta'] = np.sum(dout, axis=0)
        
        # Gradient for normalized inputs
        dx_norm = dout * self.params['gamma']
        
        # Gradient for variance
        dvar = np.sum(dx_norm * (self.inputs - self.batch_mean) * -0.5 * 
                      np.power(self.batch_var + self.epsilon, -1.5), axis=0)
        
        # Gradient for mean
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0)
        dmean += dvar * np.sum(-2 * (self.inputs - self.batch_mean), axis=0) / batch_size
        
        # Gradient for inputs
        dinputs = dx_norm / np.sqrt(self.batch_var + self.epsilon)
        dinputs += dvar * 2 * (self.inputs - self.batch_mean) / batch_size
        dinputs += dmean / batch_size
        
        return dinputs
