# need to determine useful test cases for this

import cy_fast_glcm
import numpy as np

sz = 5
a = np.random.randint(low=0, high=127, size=(200, 200, sz, sz, sz), dtype=np.uint8)
b = np.random.randint(low=0, high=127, size=(200, 200, sz, sz, sz), dtype=np.uint8)

import timing

contrast, correlation, asm, mean, var, glcm, mean_i, mean_j, var_i, var_j = cy_fast_glcm.cy_fast_glcm(a, b, True)
# print("Contrast: \n", contrast)
# print("Correlation: \n", correlation)
# print("ASM: \n", asm)
# print("Mean: \n", mean)
# print("Var: \n", var)
# print("GLCM: \n", glcm)
# print("Mean i: \n", mean_i)
# print("Mean j: \n", mean_j)
# print("Var i: \n", var_i)
# print("Var j: \n", var_j)
# print(np.count_nonzero(glcm))
