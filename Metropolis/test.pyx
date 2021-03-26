#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:28:51 2021

@author: artemstopnevich
"""


#from numpy.random import rand
#cimport cython
import cython
#import numpy as np
#cimport numpy as np
#import matplotlib.pyplot as plt
from libc.math cimport sqrt, exp #raw c funcs
cdef extern from "limits.h":
    int RAND_MAX # specifiying max random number 
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

srand48(2)
print(drand48())

