def convolve2d(input_matrix, kernel, padding='valid', stride=(1, 1)):
    """
    Performs a 2D convolution operation on a given input matrix with a specified kernel.

    Parameters:
        input_matrix (list of lists of floats): The input 2D matrix (image).
        kernel (list of lists of floats): The convolution kernel (filter).
        padding (str): 'valid' for no padding, 'same' to pad so output size matches input size.
        stride (tuple of ints): The stride (step sizes) in the y and x directions.

    Returns:
        output_matrix (list of lists of floats): The result of the convolution operation.
    """
    import math

    input_height = len(input_matrix)
    input_width = len(input_matrix[0])
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    stride_y, stride_x = stride

    # Flip the kernel for convolution
    kernel_flipped = [[kernel[kernel_height - 1 - m][kernel_width - 1 - n] 
                       for n in range(kernel_width)] for m in range(kernel_height)]

    # Determine padding
    if padding == 'same':
        pad_height = ((input_height - 1) * stride_y + kernel_height - input_height) // 2
        pad_width = ((input_width - 1) * stride_x + kernel_width - input_width) // 2
    elif padding == 'valid':
        pad_height = 0
        pad_width = 0
    else:
        raise ValueError("Padding must be 'valid' or 'same'")

    # Pad the input matrix
    padded_input = []
    for _ in range(pad_height):
        padded_input.append([0] * (input_width + 2 * pad_width))
    for row in input_matrix:
        padded_row = [0] * pad_width + row + [0] * pad_width
        padded_input.append(padded_row)
    for _ in range(pad_height):
        padded_input.append([0] * (input_width + 2 * pad_width))

    padded_height = len(padded_input)
    padded_width = len(padded_input[0])

    # Calculate output dimensions
    out_height = ((padded_height - kernel_height) // stride_y) + 1
    out_width = ((padded_width - kernel_width) // stride_x) + 1

    output_matrix = [[0] * out_width for _ in range(out_height)]

    # Convolution operation
    for i in range(out_height):
        for j in range(out_width):
            sum = 0
            for m in range(kernel_height):
                for n in range(kernel_width):
                    input_i = i * stride_y + m
                    input_j = j * stride_x + n
                    sum += padded_input[input_i][input_j] * kernel_flipped[m][n]
            output_matrix[i][j] = sum

    return output_matrix

# Example usage
if __name__ == "__main__":
    # Input matrix (image)
    input_matrix = [
        [1, 2, 0, 3],
        [4, 5, 6, 7],
        [0, 8, 9, 1],
        [1, 3, 5, 2]
    ]

    # Convolution kernel (filter)
    kernel = [
        [1, 0],
        [0, -1]
    ]

    # Perform convolution
    output = convolve2d(input_matrix, kernel, padding='same', stride=(1, 1))

    print("Output Matrix:")
    for row in output:
        print(row)
