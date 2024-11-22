import numpy as np

def convolve2d_numpy(input_matrix, kernel, padding='valid', stride=(1, 1)):
    """
    Performs a 2D convolution operation using NumPy.

    Parameters:
        input_matrix (ndarray): The input 2D matrix (image).
        kernel (ndarray): The convolution kernel (filter).
        padding (str): 'valid' or 'same'.
        stride (tuple of ints): The stride (step sizes) in the y and x directions.

    Returns:
        output_matrix (ndarray): The result of the convolution operation.
    """
    input_matrix = np.array(input_matrix)
    kernel = np.array(kernel)
    stride_y, stride_x = stride

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    if padding == 'same':
        pad_height = ((input_height - 1) * stride_y + kernel_height - input_height) // 2
        pad_width = ((input_width - 1) * stride_x + kernel_width - input_width) // 2
    elif padding == 'valid':
        pad_height = 0
        pad_width = 0
    else:
        raise ValueError("Padding must be 'valid' or 'same'")

    # Pad the input matrix
    padded_input = np.pad(input_matrix, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Calculate output dimensions
    out_height = ((padded_input.shape[0] - kernel_height) // stride_y) + 1
    out_width = ((padded_input.shape[1] - kernel_width) // stride_x) + 1

    output_matrix = np.zeros((out_height, out_width))

    # Flip the kernel for convolution
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Convolution operation
    for i in range(0, out_height):
        for j in range(0, out_width):
            y_start = i * stride_y
            y_end = y_start + kernel_height
            x_start = j * stride_x
            x_end = x_start + kernel_width

            region = padded_input[y_start:y_end, x_start:x_end]
            sum = np.sum(region * kernel_flipped)
            output_matrix[i, j] = sum

    return output_matrix

# Example usage
if __name__ == "__main__":
    # Input matrix (image)
    input_matrix = np.array([
        [1, 2, 0, 3],
        [4, 5, 6, 7],
        [0, 8, 9, 1],
        [1, 3, 5, 2]
    ])

    # Convolution kernel (filter)
    kernel = np.array([
        [1, 0],
        [0, -1]
    ])

    # Perform convolution
    output = convolve2d_numpy(input_matrix, kernel, padding='same', stride=(1, 1))

    print("Output Matrix:")
    print(output)
