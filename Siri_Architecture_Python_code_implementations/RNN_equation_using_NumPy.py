import numpy as np

def rnn_step(x_t, h_prev, W_ih, W_hh, b_h, activation='tanh'):
    """
    Performs one time-step of a simple RNN.

    Parameters:
    - x_t: Input vector at time t (numpy array of shape [input_size])
    - h_prev: Hidden state from the previous time step (numpy array of shape [hidden_size])
    - W_ih: Weight matrix from input to hidden (numpy array of shape [hidden_size, input_size])
    - W_hh: Weight matrix from hidden to hidden (numpy array of shape [hidden_size, hidden_size])
    - b_h: Bias vector (numpy array of shape [hidden_size])
    - activation: Activation function to use ('tanh', 'relu', 'sigmoid', or 'none')

    Returns:
    - h_t: Hidden state at current time step t (numpy array of shape [hidden_size])
    """
    # Ensure inputs are numpy arrays
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    W_ih = np.asarray(W_ih)
    W_hh = np.asarray(W_hh)
    b_h = np.asarray(b_h)

    # Check dimensions to avoid mismatches
    if x_t.shape[0] != W_ih.shape[1]:
        raise ValueError(f"Input vector x_t has incorrect size {x_t.shape[0]}, expected {W_ih.shape[1]}")
    if h_prev.shape[0] != W_hh.shape[1]:
        raise ValueError(f"Previous hidden state h_prev has incorrect size {h_prev.shape[0]}, expected {W_hh.shape[1]}")
    if W_ih.shape[0] != W_hh.shape[0] or W_ih.shape[0] != b_h.shape[0]:
        raise ValueError("Inconsistent hidden size in weight matrices and bias vector")

    # Compute the raw hidden state
    raw_hidden = np.dot(W_ih, x_t) + np.dot(W_hh, h_prev) + b_h

    # Apply activation function
    if activation == 'tanh':
        h_t = np.tanh(raw_hidden)
    elif activation == 'relu':
        h_t = np.maximum(0, raw_hidden)
    elif activation == 'sigmoid':
        # Prevent overflow
        raw_hidden = np.clip(raw_hidden, -500, 500)
        h_t = 1 / (1 + np.exp(-raw_hidden))
    elif activation == 'none':
        h_t = raw_hidden
    else:
        raise ValueError(f"Unsupported activation function '{activation}'")

    return h_t

# Example usage
if __name__ == "__main__":
    # Define dimensions
    input_size = 4
    hidden_size = 3

    # Initialize input vector x_t
    x_t = np.random.randn(input_size)

    # Initialize previous hidden state h_prev (e.g., zeros)
    h_prev = np.zeros(hidden_size)

    # Initialize weight matrices and bias vector
    W_ih = np.random.randn(hidden_size, input_size)
    W_hh = np.random.randn(hidden_size, hidden_size)
    b_h = np.random.randn(hidden_size)

    # Perform RNN computation for one time step
    h_t = rnn_step(x_t, h_prev, W_ih, W_hh, b_h, activation='tanh')

    print("Input x_t:", x_t)
    print("Previous hidden state h_prev:", h_prev)
    print("Current hidden state h_t:", h_t)
