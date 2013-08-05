from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(name='yart.loss',
        sources=['yart/loss.pyx', 'liblbfgs/lib/lbfgs.c'],
        include_dirs=['liblbfgs/include', 'liblbfgs/lib', numpy.get_include()])
]

setup(name='yart',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    packages=['yart'])
