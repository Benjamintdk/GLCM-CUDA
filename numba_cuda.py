from numba import cuda
import numpy as np
import math


@cuda.jit
def gen_glcm(A: np.ndarray, B: np.ndarray, glcm: np.ndarray):
    """ Fast GLCM Calculation in Numba CUDA. Inspired by Wang Jifei's Fast GLCM code.
    This MUST be implemented within controlled environments in Fast GLCM, else
    certain assumptions are not held.
    Windows must be 5 dims.
    (1) Window Row (2) Window Col (3) Cell Row (4) Cell Col (5) Channel
    Window values should not exceed uint16 size.
    The result array should be given.
    This is to make it easier to do Cython
    (1) Combination Array (2) Window Row (3) Window Col (4) Channel
    Result values should not exceed uint8 size.
    This will rarely happen, almost negligible.
    Results will only exceed if the window size > 255 and all values happen to be the same.
    """
    win_rows, win_cols, cell_rows, cell_cols, channels = A.shape
    x, y = cuda.grid(2)
    if x < win_rows and y < win_cols:
        for ch in range(channels):
            for cr in range(cell_rows):
                for cc in range(cell_cols):
                    i = A[x, y, cr, cc, ch]
                    j = B[x, y, cr, cc, ch]
                    glcm[i, j, x, y, ch] += 1
                    glcm[j, i, x, y, ch] += 1


@cuda.jit
def gen_contrast_asm(glcm: np.ndarray, contrast: np.ndarray, asm: np.ndarray, mean_i: np.ndarray, mean_j: np.ndarray):
    glcm_max_val, _, win_rows, win_cols, channels = glcm.shape
    x, y, z = cuda.grid(3)
    if x < win_rows and y < win_cols and z < channels:
        for i in range(glcm_max_val):
            for j in range(glcm_max_val):
                glcm_val = glcm[i, j, x, y, z] / glcm_max_val
                if glcm_val != 0:
                    contrast[x, y, z] += glcm_val * ((i - j) ** 2)
                    asm[x, y, z] += glcm_val ** 2
                    mean_i[x, y, z] += glcm_val * i
                    mean_j[x, y, z] += glcm_val * j

@cuda.jit
def gen_var_i_j(glcm: np.ndarray, mean_i: np.ndarray, mean_j: np.ndarray, var_i: np.ndarray, var_j: np.ndarray):
    glcm_max_val, _, win_rows, win_cols, channels = glcm.shape
    x, y, z = cuda.grid(3)
    if x < win_rows and y < win_cols and z < channels:
        for i in range(glcm_max_val):
            for j in range(glcm_max_val):
                glcm_val = glcm[i, j, x, y, z] / glcm_max_val
                mean_i_val = mean_i[x, y, z]
                mean_j_val = mean_j[x, y, z]
                if glcm_val != 0:
                    var_i[x, y, z] += glcm_val * (i - mean_i_val) ** 2
                    var_j[x, y, z] += glcm_val * (j - mean_j_val) ** 2

@cuda.jit
def gen_corr_mean_var(glcm: np.ndarray, mean_i: np.ndarray, mean_j: np.ndarray, var_i: np.ndarray, var_j: np.ndarray, corr: np.ndarray, mean: np.ndarray, var: np.ndarray):
    glcm_max_val, _, win_rows, win_cols, channels = glcm.shape
    x, y, z = cuda.grid(3)
    if x < win_rows and y < win_cols and z < channels:
        for i in range(glcm_max_val):
            for j in range(glcm_max_val):
                glcm_val = glcm[i, j, x, y, z] / glcm_max_val
                mean_i_val = mean_i[x, y, z]
                mean_j_val = mean_j[x, y, z]
                var_i_val = var_i[x, y, z]
                var_j_val = var_j[x, y, z]
                if glcm_val != 0 and var_i_val != 0 and var_j_val != 0:
                    corr[x, y, z] += glcm_val * ((i - mean_i_val) * (j - mean_j_val) / math.sqrt(var_i_val * var_j_val))
                mean[x, y, z] = (mean_i_val + mean_j_val) / 2
                var[x, y, z] = (var_i_val + var_j_val) / 2


import timing

sz = 5
a = np.random.randint(low=0, high=127, size=(200, 200, sz, sz, sz), dtype=np.uint8)
b = np.random.randint(low=0, high=127, size=(200, 200, sz, sz, sz), dtype=np.uint8)
max_val = np.max([a, b]) + 1
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
glcm_gpu = cuda.to_device(np.zeros((max_val, max_val, sz, sz, sz), dtype=np.uint8))
asm_gpu, contrast_gpu = cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8)), cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8))
mean_i, mean_j = cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8)), cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8))
var_i, var_j = cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8)), cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8))
corr, mean, var = cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8)), cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8)), cuda.to_device(np.zeros((sz, sz, sz), dtype=np.uint8))
threads_per_block = (8, 8)
blockspergrid_x = int(np.ceil(glcm_gpu.shape[0] / threads_per_block[0]))
blockspergrid_y = int(np.ceil(glcm_gpu.shape[1] / threads_per_block[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

gen_glcm[blockspergrid, threads_per_block](a_gpu, b_gpu, glcm_gpu)
gen_contrast_asm[blockspergrid, threads_per_block](glcm_gpu, contrast_gpu, asm_gpu, mean_i, mean_j)
gen_var_i_j[blockspergrid, threads_per_block](glcm_gpu, mean_i, mean_j, var_i, var_j)
gen_corr_mean_var[blockspergrid, threads_per_block](glcm_gpu, mean_i, mean_j, var_i, var_j, corr, mean, var)
cuda.synchronize()
cuda_glcm = glcm_gpu.copy_to_host()
cuda_asm = asm_gpu.copy_to_host()
cuda_contrast = contrast_gpu.copy_to_host()
cuda_mean_i = mean_i.copy_to_host()
cuda_mean_j = mean_j.copy_to_host()
cuda_var_i = var_i.copy_to_host()
cuda_var_j = var_j.copy_to_host()
cuda_corr = corr.copy_to_host()
cuda_mean = mean.copy_to_host()
cuda_var = var.copy_to_host()
