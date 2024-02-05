# import the torch library to code using pytorch
import torch as tc; # semi colons are not required, it is a habit developed while using languages that do require semi colons at the enbd of every line


# Scalars are really no different from regular varible with single vaslues
scalarEX = tc.tensor(1)

# arrays are variables that contain multiple different value that can be called upon individualy
arrayEX = tc.tensor(
    [1,2,3,4,5]
    )

# Matrices are 2d arrays
matrixEX = tc.tensor(
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
    )

# Tensors are 3d arrays
tensorEX = tc.tensor(
    [[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
    [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]]
    )