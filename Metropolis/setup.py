#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:37:07 2021

@author: artemstopnevich
"""

import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
from distutils.extension import Extension
from Cython.Distutils import build_ext


def checkOpenmpSupport():
    """ Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    """ 
    ompTest = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Thread %d, Total number of threads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    with open(filename, 'w') as file:
        file.write(ompTest)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call(['cc', '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)

    os.chdir(curdir)
    shutil.rmtree(tmpdir) 
    if result == 0:
        return True
    else:
        return False

if checkOpenmpSupport() == True:
    ompArgs = ['-fopenmp']
else:
    ompArgs = None 
    
    
ext_modules=[
    Extension("ising",
              sources=['core_ising/ising.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=ompArgs,
              extra_link_args=ompArgs)
]

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
  )


'''
run setup file in the terminal:
    
    python setup.py build_ext --inplace
'''



