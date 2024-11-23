import numpy as np

def attention(Q, K, V, mask=None):
    """
    Computes the scaled dot-product attention.

    Parameters:
    - Q (ndarray): Query matrix of shape (batch_size, seq_len_q, d_k)
    - K (ndarray): Key matrix of shape (batch_size, seq_len_k, d_k)
    - V (ndarray): Value matrix of shape (batch_size, seq_len_k, d_v)
    - mask (ndarray, optional): Mask matrix broadcastable to (batch_size, seq_len_q, seq_len_k)

    Returns:
    - output (ndarray): Attention output of shape (batch_size, seq_len_q, d_v)
    - attention_weights (ndarray): Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    # Ensure the dimensions are compatible
    batch_size, seq_len_q, d_k = Q.shape
    _, seq_len_k, _ = K.shape
    _, seq_len_v, d_v = V.shape

    assert K.shape[0] == batch_size and V.shape[0] == batch_size, "Batch sizes of K and V must match Q"
    assert K.shape[2] == d_k, f"Key dimension {K.shape[2]} must match Query dimension {d_k}"
    assert seq_len_k == seq_len_v, f"Sequence lengths of K {seq_len_k} and V {seq_len_v} must match"

    # Step 1: Compute raw attention scores by matrix multiplication between Q and K^T
    # Resulting shape: (batch_size, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # Step 2: Scale the scores by square root of key dimension for stability
    # Avoid division by zero in case d_k is zero
    if d_k == 0:
        raise ValueError("Key dimension d_k must be greater than zero")
    scaling_factor = np.sqrt(d_k)
    scores = scores / scaling_factor

    # Step 3: Apply the mask (if provided)
    if mask is not None:
        if mask.shape != scores.shape:
            # Attempt to broadcast the mask
            try:
                mask = np.broadcast_to(mask, scores.shape)
            except ValueError:
                raise ValueError(f"Mask shape {mask.shape} cannot be broadcast to scores shape {scores.shape}")
        # Use a large negative value instead of -inf to avoid NaNs during backprop
        scores = np.where(mask, scores, -1e9)

    # Step 4: Compute the softmax of the scores in a numerically stable way
    # Subtract the max for numerical stability (softmax(x) = softmax(x - max(x)))
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)

    # Edge case: if all scores are -inf (masked), exp_scores will be zeros leading to division by zero
    sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
    # To avoid division by zero, set zeros to ones (since exp(-inf) = 0)
    sum_exp_scores = np.where(sum_exp_scores == 0, 1, sum_exp_scores)

    attention_weights = exp_scores / sum_exp_scores  # Shape: (batch_size, seq_len_q, seq_len_k)

    # Step 5: Compute the output by multiplying attention weights with V
    output = np.matmul(attention_weights, V)  # Shape: (batch_size, seq_len_q, d_v)

    # Edge case: Check for NaNs or Infs in outputs
    if np.isnan(output).any() or np.isinf(output).any():
        raise FloatingPointError("NaN or Inf detected in the output")

    return output, attention_weights

# Example usage:
if __name__ == "__main__":
    # Define dimensions
    batch_size = 2
    seq_len_q = 3
    seq_len_k = seq_len_v = 4
    d_k = 64
    d_v = 64

    # Randomly initialize Q, K, V matrices
    Q = np.random.rand(batch_size, seq_len_q, d_k)
    K = np.random.rand(batch_size, seq_len_k, d_k)
    V = np.random.rand(batch_size, seq_len_v, d_v)

    # Optional mask (e.g., padding mask)
    mask = np.random.randint(0, 2, size=(batch_size, seq_len_q, seq_len_k)).astype(bool)

    # Compute attention
    output, attention_weights = attention(Q, K, V, mask)

    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)
