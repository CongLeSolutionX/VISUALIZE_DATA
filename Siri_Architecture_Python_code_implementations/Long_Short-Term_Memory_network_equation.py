import numpy as np

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        """
        Initializes the LSTM cell with given dimensions.

        Parameters:
        - input_dim: Dimension of the input vector x_t.
        - hidden_dim: Dimension of the hidden state h_t and cell state C_t.
        - output_dim: Dimension of the output. If None, output_dim = hidden_dim.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim

        # Weight matrices for input, forget, cell, and output gates
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))

        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))

        self.W_C = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b_C = np.zeros((hidden_dim, 1))

        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))

        # For the output layer (if output_dim != hidden_dim)
        if self.output_dim != self.hidden_dim:
            self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(1. / output_dim)
            self.b_y = np.zeros((output_dim, 1))
        else:
            self.W_y = None
            self.b_y = None

    def sigmoid(self, z):
        """
        Numerically stable sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, x_t, h_prev, C_prev):
        """
        Forward pass of the LSTM cell.

        Parameters:
        - x_t: Input at time t, shape (input_dim, batch_size)
        - h_prev: Previous hidden state, shape (hidden_dim, batch_size)
        - C_prev: Previous cell state, shape (hidden_dim, batch_size)

        Returns:
        - h_t: Current hidden state, shape (hidden_dim, batch_size)
        - C_t: Current cell state, shape (hidden_dim, batch_size)
        - y_t: Output (if output layer is defined), shape (output_dim, batch_size)
        """
        # Ensure inputs are two-dimensional
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if C_prev.ndim == 1:
            C_prev = C_prev.reshape(-1, 1)

        # Concatenate h_prev and x_t
        concat = np.vstack((h_prev, x_t))  # Shape: (hidden_dim + input_dim, batch_size)

        # Forget Gate
        f_t = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
        # Input Gate
        i_t = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
        # Candidate Memory Cell
        C_hat_t = np.tanh(np.dot(self.W_C, concat) + self.b_C)
        # Cell State Update
        C_t = f_t * C_prev + i_t * C_hat_t
        # Output Gate
        o_t = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
        # Hidden State Update
        h_t = o_t * np.tanh(C_t)

        # Output Layer (if applicable)
        if self.W_y is not None:
            y_t = np.dot(self.W_y, h_t) + self.b_y
        else:
            y_t = h_t  # If no separate output layer, hidden state is the output

        return h_t, C_t, y_t

    def init_hidden_state(self, batch_size=1):
        """
        Initializes the hidden state and cell state to zeros.

        Parameters:
        - batch_size: Number of sequences to process in parallel.

        Returns:
        - h_0: Initial hidden state, shape (hidden_dim, batch_size)
        - C_0: Initial cell state, shape (hidden_dim, batch_size)
        """
        h_0 = np.zeros((self.hidden_dim, batch_size))
        C_0 = np.zeros((self.hidden_dim, batch_size))
        return h_0, C_0

# Example usage
if __name__ == "__main__":
    # Define dimensions
    input_dim = 10    # Dimension of input vector x_t
    hidden_dim = 20   # Dimension of hidden state h_t and cell state C_t
    output_dim = 5    # Dimension of the output y_t

    # Create an LSTM cell instance
    lstm_cell = LSTMCell(input_dim, hidden_dim, output_dim)

    # Initialize hidden state and cell state
    batch_size = 3  # For example, processing 3 sequences in parallel
    h_prev, C_prev = lstm_cell.init_hidden_state(batch_size)

    # Example input at time t
    x_t = np.random.randn(input_dim, batch_size)

    # Forward pass
    h_t, C_t, y_t = lstm_cell.forward(x_t, h_prev, C_prev)

    print("Hidden State h_t shape:", h_t.shape)
    print("Cell State C_t shape:", C_t.shape)
    print("Output y_t shape:", y_t.shape)

