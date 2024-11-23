import torch
import torch.nn as nn
import math

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation='tanh'):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        
        # Define parameters
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Select activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function '{activation}'")
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)
    
    def forward(self, x_t, h_prev):
        # x_t: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)
        
        # Compute x_t @ W_ih^T
        x_W_ih_T = torch.matmul(x_t, self.W_ih.t())  # Shape: (batch_size, hidden_size)
        
        # Compute h_prev @ W_hh^T
        h_prev_W_hh_T = torch.matmul(h_prev, self.W_hh.t())  # Shape: (batch_size, hidden_size)
        
        # Sum and add bias
        raw_hidden = x_W_ih_T + h_prev_W_hh_T + self.b_h  # Shape: (batch_size, hidden_size)
        
        # Apply activation function
        h_t = self.activation(raw_hidden)
        return h_t

# Example usage
if __name__ == "__main__":
    input_size = 4
    hidden_size = 3
    batch_size = 1

    # Instantiate the RNN cell
    rnn_cell = CustomRNNCell(input_size, hidden_size, activation='tanh')

    # Create input and hidden state tensors
    x_t = torch.randn(batch_size, input_size)
    h_prev = torch.zeros(batch_size, hidden_size)

    # Compute the next hidden state
    h_t = rnn_cell(x_t, h_prev)

    print("Next hidden state h_t:", h_t)
