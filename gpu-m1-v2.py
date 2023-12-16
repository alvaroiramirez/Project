import torch
import platform


# Create a tensor on the mps device
x = torch.ones(1000, device="mps")

# Perform an operation on the tensor
y = torch.sum(x)

# Print the result
print(y)



# Get the platform
platform_name = platform.system()

# Detect the GPU
if platform_name == "Darwin":  # macOS
    device = torch.device("mps" if torch.has_mps() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device name
print(device)

# Perform a simple operation on the GPU
if device == "mps" or device == "cuda":
    x = torch.ones(1000, device=device)
    y = torch.sum(x)
    print(y)

# If no GPU is available, print a message
else:
    print("No GPU available.")
