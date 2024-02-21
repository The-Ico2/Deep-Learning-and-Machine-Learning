import torch as tc;

tensor1 = tc.tensor([1,2,3]) # tc.randn(height, width)
tensor2 = tc.tensor([2,3,4]) # tc.randn(height, width)
print("Original1: ", tensor1)
print("Original2: ", tensor2, "\n")

# adding to a tensor
print("Adding: ", tensor1 + 10)

# subtracting from a tensor
print("subtracting: ", tensor1 - 5)

# dividing a tensor
print("dividing: ", tensor1 / 9)

# multiplying a tensor
print("multiplying: ", tensor1 * 2, "\n")

# You can also add, multiply, divide, and subtract one array from another
# Note, the tensors your doing this to have to be the EXACT SAME size (height, length, width)
print("tensor to tensor: adding, subtracting, dividing, and multiplying")
print("Adding: ",tensor1 + tensor2)
print("subtracting: ",tensor1 - tensor2)
print("dividing: ",tensor1 / tensor2, "\n")
print("Multiplication: ")
print(tensor1 * tensor2) # Element-Wise = [2, 6, 12]
print(tc.matmul(tensor1,tensor2)) # Matrix Multiplication / Dot product 2+6+12 = 20
# matmul only works if the tensors are the same shape




# Transpose switches the axis or given dimension of a given tensor; .T is the function for transposing
tensor_A = tc.randn(3,2)
tensor_B = tc.randn(3,2)
print(tc.mm(tensor_A, tensor_B.T))
#test