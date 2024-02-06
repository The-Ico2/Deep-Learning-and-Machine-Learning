import torch as tc;

randTensor = tc.randn(2, 5) # tc.randn(height, width)
print(randTensor, "\n", randTensor.dtype, "\n")

#Note, These can only be used when the variable contains a tensor
print(randTensor.min())
print(randTensor.max())
print(randTensor.sum())
print(randTensor.mean()) # requires datatype to be float32 to perform

print(randTensor.argmin()) # Finds the index position of the smallest value
print(randTensor.argmax()) # Finds the index position of the largest value
print("Nutttttttttttttt")