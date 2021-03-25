#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:00:35 2021

@author: artemstopnevich
"""

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

'''
    Series of functions which will extract all major order parameters
    from the Ising simulation
'''

def magnetization(lattice):
    ''' calculate m'''
    Ns = np.prod(lattice.shape)
    return abs((1/Ns) * np.sum(lattice));

def energy(lattice, J=1):
    ''' calculate e '''
    N = lattice.shape;
    dimension = len(N); #trying to go dimensionless
    NN = copy(lattice);
    E = 0;
    neighbours = 0;
    for i in range(dimension):
        for j in [-1,1]:
            neighbours += np.roll(NN, shift=j, axis = i);
            E+=J*np.sum(lattice*np.roll(NN, shift=j, axis = i));
    DeltaE = J * (lattice* neighbours)/(np.prod(N));
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
    chi = beta*(M2 - M*M);
    return chi;

def heat_capacity(E, E2, beta):
    ''' formula cv = (<e^2> - <e>^2)*(J/T)^2 '''
    cv = (E2 - E*E)*(beta**2);
    return cv;

def binder(M, M2):
    Q = M2/(M*M);
    return Q


def observables_fig(Temperature, data, errors, L, epochs):
    f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
    f.suptitle("Observables of {}x{} lattice with {} iterations".format(L,L,epochs), fontsize=20);
    sp =  f.add_subplot(2, 2, 1 );
    plt.plot(Temperature, data.ene, 'o', color="#A60628", label=' Energy');
    plt.errorbar(Temperature, data.ene, fmt='none', xerr=0, yerr= errors.ene);
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);
    
    sp =  f.add_subplot(2, 2, 2 );
    plt.plot(Temperature, data.mag, '*', color="#348ABD", label='Magnetization');
    plt.errorbar(Temperature, data.mag, fmt='none', xerr=0, yerr= errors.mag)
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Magnetization ", fontsize=20);
    
    
    sp =  f.add_subplot(2, 2, 3 );
    plt.plot(Temperature, data.sp_heat, 'd', color='black', label='Specific Heat');
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);
    
    
    sp =  f.add_subplot(2, 2, 4 );
    plt.plot(Temperature, data.susc, 's', label='Susceptebility');
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Suseptibility", fontsize=20);
    
    plt.savefig("figures/observables_{}_grid_{}_steps.png".format(L, epochs))
    plt.show()
    
def observables_fig_Metropolis(T, E, M, C, X, EE, EM):
    f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    

    sp =  f.add_subplot(2, 2, 1 );
    plt.plot(T, E, 'o', color="#A60628", label=' Energy');
    plt.errorbar(T, E, fmt='none', xerr=0, yerr= EE);
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20);
    
    sp =  f.add_subplot(2, 2, 2 );
    plt.plot(T, abs(M), '*', color="#348ABD", label='Magnetization');
    plt.errorbar(T, abs(M), fmt='none', xerr=0, yerr= EM)
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Magnetization ", fontsize=20);


    sp =  f.add_subplot(2, 2, 3 );
    plt.plot(T, C, 'd', color='black', label='Specific Heat');
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Specific Heat ", fontsize=20);
    
    
    sp =  f.add_subplot(2, 2, 4 );
    plt.plot(T, X, 's', label='Specific Heat');
    plt.legend(loc='best', fontsize=15); 
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Suseptibility", fontsize=20);
    
    
    plt.show()
    
# import math 
# M_T[t] = pow(1- pow(np.sinh(2*iT), -4), 1/8)