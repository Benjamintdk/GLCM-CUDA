from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cy_fast_glcm.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)