from numba_cuda import gen_glcm, gen_contrast_asm, gen_var_i_j, gen_corr_mean_var
from numba import cuda
import cy_fast_glcm
import numpy as np


def test_result():
    sz = 2
    # a = np.random.randint(low=0, high=127, size=(2, 2, 2, 2, 2), dtype=np.uint8)
    # b = np.random.randint(low=0, high=127, size=(2, 2, 2, 2, 2), dtype=np.uint8)
    # a = np.ones((sz, sz, sz, sz, sz)).astype('uint8')
    # b = np.ones((sz, sz, sz, sz, sz)).astype('uint8')
    a = np.full((sz, sz, sz, sz, sz), 126, dtype=np.uint8)
    b = np.full((sz, sz, sz, sz, sz), 126, dtype=np.uint8)
    max_val = np.max([a, b]) + 1
    cy_contrast, cy_correlation, cy_asm, cy_mean, cy_var, cy_glcm, cy_mean_i, cy_mean_j, cy_var_i, cy_var_j = cy_fast_glcm.cy_fast_glcm(a, b, True)

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
    assert(np.allclose(cy_glcm, cuda_glcm))
    assert(np.allclose(cy_asm, cuda_asm))
    assert(np.allclose(cuda_mean_i, cy_mean_i))
    assert(np.allclose(cuda_mean_j, cy_mean_j))
    assert(np.allclose(cuda_var_i, cy_var_i))
    assert(np.allclose(cuda_var_j, cy_var_j))
    assert(np.allclose(cuda_corr, cy_correlation))
    assert(np.allclose(cuda_mean, cy_mean))
    assert(np.allclose(cuda_var, cy_var))
    assert(np.allclose(cy_contrast, cuda_contrast))
    