# import the torch library to code using pytorch
import torch as tc; # semi colons are not required, it is a habit developed while using languages that do require semi colons at the enbd of every line


# Scalars are really no different from regular varible with single vaslues
scalarEX = tc.tensor(1)
print(scalarEX, "\n", scalarEX.dtype, scalarEX.shape, scalarEX.device) # Printing scalar with all information such as as data tyoe, shaoe and device

# arrays are variables that contain multiple different value that can be called upon individualy
arrayEX = tc.tensor(
    [1,2,3,4,5]
    )
print(arrayEX, "\n", arrayEX.dtype, arrayEX.shape, arrayEX.device) # Printing array with all information such as as data tyoe, shaoe and device

# Matrices are 2d arrays
matrixEX = tc.tensor(
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
    )
print(matrixEX, "\n", matrixEX.dtype, matrixEX.shape, matrixEX.device) # Printing matrix with all information such as as data tyoe, shaoe and device

# Tensors are 3d arrays
tensorEX = tc.tensor(
    [[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]]
    )
print(tensorEX, "\n", tensorEX.dtype, tensorEX.shape, tensorEX.device) # Printing tensor with all information suchas as data tyoe, shaoe and device