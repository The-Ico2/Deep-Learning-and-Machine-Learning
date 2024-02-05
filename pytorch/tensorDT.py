# import the torch library to code using pytorch
import torch as tc;


# 16 bit Float Tensor
float_16_tensor = tc.tensor([3.0, 6.0, 9.0], dtype=tc.float16)
# 32 bit Float Tensor
float_32_tensor = tc.tensor([9.0, 4.0, 5.0], dtype=tc.float32)
# 64 bit Float Tensor
float_64_tensor = tc.tensor([1.0, 17.0, 20.0], dtype=tc.float64)


# the default value set to all float point variables is 32 bit float
print(float_32_tensor.dtype)

# changing float datatype
float_16_tensorEX1 = float_32_tensor.type(tc.float16)
float_16_tensorEX2 = tc.tensor([3.0, 6.0, 9.0], dtype=tc.float16)
print(float_16_tensorEX1.dtype, "\n", float_16_tensorEX2.dtype)


# datatype after multipling tensors
float32x16 = float_32_tensor * float_16_tensor
flaot32x64 = float_32_tensor * float_64_tensor
float16x64 = float_16_tensor * float_64_tensor

print(float32x16, float32x16.dtype)
print(flaot32x64, flaot32x64.dtype)
print(float16x64, float16x64.dtype)
# When multiplting, dividing, subtracting, or adding tensors of different datatypes like float16 float32 and float64 the datatype will be set to the largest bit type