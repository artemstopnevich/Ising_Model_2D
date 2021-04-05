#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:58:16 2021

@author: artemstopnevich
"""


#from numpy.random import rand
cimport cython
import cython
import numpy as np
cimport numpy as np
from copy import copy
import matplotlib.pyplot as plt
from cython.parallel import prange
from libc.math cimport sqrt, exp #raw c funcs
from libc.stdlib cimport rand
import time


cdef extern from "time.h" :
    pass
cdef extern from "limits.h":
    int RAND_MAX # specifiying max random number 
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)



##############################################################################
# MAIN CODE
##############################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef class Ising:
    cdef int N, epochs, nt
    def __init__(self, N, nt, epochs):
        self.N      = N 
        self.nt     = nt
        self.epochs = epochs
        pass    

    cpdef twoD(self, int [:, :] Field, double [:] E, double [:] M, double [:] C, double [:] X, double [:] B, double [:] Q):
        cdef int epochs = self.epochs, N = self.N, nt = self.nt;
        cdef double E1, M1, E2, M2, beta, Mag, Ene; 
        cdef int i, j, tt, N2 = N*N;
        cdef int relax_time = 1000;
        cdef double iMCS = 1.0/(epochs-relax_time), iNs = 1.0/(N2)
        cdef long int seedval = time.time()

        srand48(seedval)
        for tt in range(nt):
            print(tt, nt);
            E1 = E2 = M1 = M2= 0;
            Mag = 0;
            Ene = 0
            beta = B[tt];
            initialise(Field, N);
                

            for i in range(relax_time):
                ising_step(Field, beta, N);
            for j in range(epochs-relax_time):
                ising_step(Field, beta, N);
                
                Ene = calcEnergy(Field, N);                 
                Mag = calcMag(Field, N);
                E1 = E1 + Ene;
                M1 = M1 + Mag;
                E2 = E2 + Ene*Ene;
                M2 = M2 + Mag*Mag;
                            
            E[tt] = E1*iMCS*iNs 
            M[tt] = M1*iMCS*iNs 
            C[tt] = (E2*iMCS - E1*E1*iMCS*iMCS)*beta*beta*iNs;
            X[tt] = (M2*iMCS - M1*M1*iMCS*iMCS)*beta*iNs;
            Q[tt] = M2*iMCS/(M1*M1*iMCS*iMCS)
            
        return 

##############################################################################
# FUNCTIONS
##############################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef int initialise(int [:,:] Field, int N):    # generates a random spin Spin configuration
    '''generates a random spin configuration for initial condition'''
    cdef int i, j
    for i in range(1, N+1):
        for j in range(1, N+1):
            if (drand48() < 0.5):
                Field[i, j]=-1
            else:
                Field[i, j]=1
    return 0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef int ising_step(int [:, :] Field, double beta, int N):
    cdef int ii, a, b, dE
    for ii in range(N*N):
        a = int(1 + drand48()*N)
        b = int(1 + drand48()*N) 

        Field[0, b]   = Field[N, b];  Field[N+1, b] = Field[1, b];  # ensuring BC
        Field[a, 0]   = Field[a, N];  Field[a, N+1] = Field[a, 1];
        
        dE = 2*Field[a, b]*(Field[a+1, b] + Field[a, b+1] + Field[a-1, b] + Field[a, b-1])
        if (dE < 0):          #have a look at C replacement
            Field[a,b] *= -1
        elif (drand48() < exp(-dE*beta)):
            Field[a,b] *= -1
    return 0
                        
                        
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef int calcEnergy(int [:, :] Field, int N) nogil:
    ''' Energy calculation'''
    cdef int i, j, energy = 0
    for i in prange(1, N+1, nogil=True):
        for j in range(1, N+1):
            Field[0, j] = Field[N, j];  Field[i, 0] = Field[i, N];
            energy += -Field[i, j] * (Field[i-1, j] + Field[i, j-1]) 
    return energy/2

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef int calcMag(int [:, :] Field, int N) nogil:
    '''Magnetization of a given configuration'''
    cdef int i, j, mag = 0
    for i in prange(1, N+1, nogil=True):
        for j in range(1, N+1):
            mag += Field[i,j]
    return mag




@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef float blocking_error(double [:] A, int num_blocks): # rewrite equation
    mean_values = np.zeros(num_blocks)
    error_block = np.zeros(num_blocks)
    itemize = range(1, num_blocks+1)
    blocks = np.array_split(A, num_blocks)
    cdef  error, global_mean = np.mean(A)
    cdef int i
    for i in range(num_blocks):
        mean_values[i] = np.mean(blocks[i])
        error_block= np.sqrt((1/itemize[i]) * np.sum((mean_values - global_mean)**2))
    error = error_block/np.sqrt(num_blocks)

    return error

