#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:00:35 2021

@author: artemstopnevich
"""

import numpy as np
from copy import copy
import math
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

'''
    Series of functions which will extract all major order parameters
    from the Ising simulation
'''

def magnetization(lattice):
    ''' calculate m'''
    N = np.prod(lattice.shape)
    return np.sum(lattice);


def energy(lattice):
    ''' calculate e '''
    N = lattice.shape;
    dimension = len(N); #trying to go dimensionless
    NN = copy(lattice);
    E = 0;
    neighbours = 0;
    for i in range(dimension):
        for j in [-1,1]:
            neighbours += np.roll(NN, shift=j, axis = i);
            E += np.sum(lattice*np.roll(NN, shift=j, axis = i));
    DeltaE = (lattice*neighbours);
    return  -np.sum(DeltaE)/2;
#    return -(E/np.prod(N))/2; #return is avg energy per site

def magnetization_2(lattice): 
    ''' calculate m^2 '''
    return (magnetization(lattice)*magnetization(lattice)); 

def energy_2(lattice,J=1):
    ''' calculate e^2 '''
    N_s = np.prod(lattice.shape);
    N = lattice.shape;
    dimension = len(N); #dimension of system being simulated
    NN = copy(lattice);
    E = 0;
    for i in range(dimension):
        for j in [-1,1]:
            E+=J*(lattice*np.roll(NN, shift=j, axis = i));
    return np.sum(E*E)/N_s/2;


def susceptibility(M, M2, beta):
    ''' formula chi = 1/T(<m^2> - <m>^2) '''
    chi = beta*(M2 - abs(M)*abs(M));
    return chi;

def specific_heat(E, E2, beta):
    ''' formula cv = (<e^2> - <e>^2)*(J/T)^2 '''
    cv = (E2 - E*E)*(beta**2);
    return cv;

def binder(M, M2):
    Q = M2/(M*M);
    return Q


def calcEnergy(Field):
    ''' Energy of a given configuration '''
    N = Field.shape[0];
    Ns = np.prod(Field.shape);
    energy = 0;
    for i in range(N):
        for j in range(N):
            Field[0, j] = Field[N-1, j];  Field[i, 0] = Field[i, N-1];
            energy += -Field[i, j] * (Field[i-1, j] + Field[i, j-1]);
    return energy/2
    
def calcMag(Field):
    ''' Magnetization of a given configuration '''
    N = Field.shape[0];
    mag = 0;
    for i in range(N):
        for j in range(N):
            mag += Field[i,j];
    return mag

def observables_fig(T, E, M, EE, EM, L, epochs,f, sp1, sp2, m, c):


    sp1.plot(T, E, m, color = c, label='{}x{}'.format(L,L), markersize = 4);
    sp1.errorbar(T, E, color=c, fmt='none', xerr=0, yerr= EE);
    sp1.set_xlabel("Temperature (T)", fontsize=20);
    sp1.set_ylabel("Energy ", fontsize=20);
    sp1.legend(loc='best', fontsize=15)
    
    sp2.plot(T, abs(M), m, color = c, label='{}x{}'.format(L,L), markersize = 4);
    sp2.errorbar(T, abs(M), color=c, fmt='none', xerr=0, yerr= EM)
    sp2.set_xlabel("Temperature (T)", fontsize=20);
    sp2.set_ylabel("Magnetization ", fontsize=20);
    sp2.legend(loc='best', fontsize=15)
    
    '''    
    sp =  f.add_subplot(2, 2, 3 );
    plt.plot(T, C, 'd', color='black', label='Specific Heat');
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);
    
    
    sp =  f.add_subplot(2, 2, 4 );
    plt.plot(T,X, 's', label='Susceptebility');
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Suseptibility", fontsize=20);
    '''    

    
def observables_fig_Metropolis(T, E, M, C, X, EE, EM, EC, EX, N, f, sp1, sp2, c, m):
    '''
    sp1.plot(T, E, m, color = c, label='{}x{}'.format(N,N), markersize = 4);
    sp1.errorbar(T, E, color=c, fmt='none', xerr=0, yerr= EE);
#    sp1.set_xlabel("Temperature (T)", fontsize=20);
    sp1.set_ylabel("Energy ", fontsize=20);
    sp1.legend(loc='best', fontsize=15)
    
    '''
    sp1.plot(T, M, m, color =c, label='N=220', markersize = 4);
    sp1.errorbar(T, M, color=c, fmt='none', xerr=0, yerr= EM)
    sp1.set_xlabel("Temperature (T)", fontsize=20);
    sp1.set_ylabel("Magnetization ", fontsize=20);
    sp1.legend(loc='best', fontsize=15);
    '''
    sp3.plot(T, C, m, color=c, label='{}x{}'.format(N,N), markersize = 4);
    sp3.errorbar(T, C, color=c, fmt='none', xerr=0, yerr= EC);
    sp3.set_xlabel("Temperature (T)", fontsize=20);
    sp3.set_ylabel("Specific Heat ", fontsize=20);
    sp3.legend(loc='best', fontsize=15)
    '''    
    sp2.plot(T, X, m, color = c, label='N=220', markersize = 4);
    sp2.errorbar(T, X, color=c, fmt='none', xerr=0, yerr= EX);
    sp2.set_xlabel("Temperature (T)", fontsize=20);
    sp2.set_ylabel("Susceptibility", fontsize=20);
    sp2.legend(loc='best', fontsize=15);    
   

    
# import math 
# M_T[t] = pow(1- pow(np.sinh(2*iT), -4), 1/8)