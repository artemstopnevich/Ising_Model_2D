#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:58:16 2021

@author: artemstopnevich
"""


from numpy.random import rand
cimport cython
import cython
import numpy as np
cimport numpy as np
from copy import copy
import matplotlib.pyplot as plt
from core_functions.physical_measures import *
from libc.math cimport sqrt, exp #raw c funcs
cdef extern from "limits.h":
    int RAND_MAX # specifiying max random number 


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

    cpdef twoD(self, np.int64_t[:, :] Field, double [:] E, double [:] M, double [:] C, double [:] X, double [:] B, double [:] Q, double [:] Error_E, double [:] Error_M):
        cdef int epochs = self.epochs, N = self.N, nt = self.nt;
        cdef double E1, M1, E2, M2, beta, Ene, Mag;
        cdef int i, j, tt, N2 = N*N;
#        cdef float [:,:] data ;
        cdef int relax_time = 300;
        cdef double iMCS = 1.0/epochs, iNs = 1.0/N2;
        initialise(Field, N);
        for tt in range(nt):
            print(tt, nt);
#            ene_array = mag_array = np.zeros((epochs-relax_time), dtype=np.float64);
            E1 = E2 = M1 = M2= 0;
            Mag = 0;
            beta = B[tt];
            data = np.zeros((epochs-relax_time,4))
            for i in range(epochs):
                ising_step(Field, beta, N);
                if i > relax_time:
                    j = i-relax_time;
                    
                    M1 = calcMag(Field, N)/(N*N)
                    E1 = calcEnergy(Field, N);
                    M2 = M1*M1;
                    E2 = E1*E1;
                    
                    data[j,:] = [M1, E1, M2, E2];
            Error_E[tt] = blocking_error(data[:,1], 10);
            Error_M[tt] = blocking_error(data[:,0], 10);
            
            data = np.mean(data, axis=0);
            
            M[tt] = data[0];      
            E[tt] = data[1];
            C[tt] = (data[3] - data[1]**2)*beta**2;
            X[tt] = (data[2] - data[0]**2)*beta;
            
            Q[tt] = data[2]/(data[0]**2);

        return 

##############################################################################
# FUNCTIONS
##############################################################################
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef initialise(np.int64_t[:,:] Field, int N):    # generates a random spin Spin configuration
    '''generates a random spin configuration for initial condition'''
    cdef int i, j
    for i in range(1, N+1):
        for j in range(1, N+1):
            if rand() < 0.5: 
                Field[i, j]=-1
            else:
                Field[i, j]=1
    return 0 

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef ising_step(np.int64_t[:, :] Field, float beta, int N):
    cdef int ii, a, b
    for ii in range(N*N):
        a = np.random.randint(0,N)
        b = np.random.randint(0,N)
        Field[0, b]   = Field[N, b];  Field[N+1, b] = Field[1, b];  # ensuring BC
        Field[a, 0]   = Field[a, N];  Field[a, N+1] = Field[a, 1];
        
        dE = 2*Field[a, b]*(Field[a+1, b] + Field[a, b+1] + Field[a-1, b] + Field[a, b-1])
        if dE < 0:          #have a look at C replacement
            Field[a,b] *= -1
        elif rand() < exp(-dE*beta):
            Field[a,b] *= -1
    return 0

        
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef float calcEnergy(np.int64_t[:, :] Field, int N):
    ''' Energy calculation'''
    cdef int i, j, energy = 0
    for i in range(1, N+1):
        for j in range(1, N+1):
            Field[0, j] = Field[N, j];  Field[i, 0] = Field[i, N];
            energy += -Field[i, j] * (Field[i-1, j] + Field[i, j-1]) 
    return energy/(N*N)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef float calcMag(np.int64_t[:,:] Field, int N):
    '''Magnetization of a given configuration'''
    cdef int i, j, mag = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            mag += Field[i,j]
    return mag

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef float blocking_error(double [:] A, int num_blocks):
    mean_values = np.zeros(num_blocks)
    error_block = np.zeros(num_blocks)
    itemize = range(1, num_blocks+1)
    blocks = np.array_split(A, num_blocks)
    cdef  error, global_mean = np.mean(A)
    cdef int i
    for i in range(num_blocks):
        mean_values[i] = np.mean(blocks[i])
        error_block = np.sqrt((1/itemize[i]) * np.sum((mean_values - global_mean)**2))
    error = error_block/np.sqrt(num_blocks)

    return error

