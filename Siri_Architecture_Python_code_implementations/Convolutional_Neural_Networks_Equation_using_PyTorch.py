import torch
import torch.nn.functional as F

# Input and kernel tensors
input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Perform convolution
output_tensor = F.conv2d(input_tensor, kernel_tensor, padding=1)

print("Output Matrix:")
print(output_tensor.squeeze().numpy())
