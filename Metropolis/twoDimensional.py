#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:26:55 2021

@author: artemstopnevich
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pyximport
pyximport.install()
import ising

##############################################################################
# SIMULATION
##############################################################################
N = 16
nt = 30
eqSteps = 1000
mcSteps = 1000

Field           = np.zeros((N+2,N+2), dtype=np.int64)
Energy          = np.zeros((nt), dtype=np.float64)
Magnetization   = np.zeros((nt), dtype=np.float64)
SpecificHeat    = np.zeros((nt), dtype=np.float64)
Susceptibility  = np.zeros((nt), dtype=np.float64)
Error_Ene       = np.zeros((nt), dtype=np.float64)
Error_Mag       = np.zeros((nt), dtype=np.float64)

#Temperature
Temperature = np.linspace(0.5,3.5, nt)
Beta = 1.0/Temperature   # set k_B = 1

Ising = ising.Ising(N, nt, eqSteps, mcSteps)

Ising.twoD(Field, Energy, Magnetization, SpecificHeat, Susceptibility, Beta, Error_Ene, Error_Mag)


##############################################################################
# VISUALIZATION
##############################################################################
f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    

sp =  f.add_subplot(2, 2, 1 );
plt.plot(Temperature, Energy, 'o', color="#A60628", label=' Energy');
plt.errorbar(Temperature, Energy, fmt='none', xerr=0, yerr= Error_Ene);
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);

sp =  f.add_subplot(2, 2, 2 );
plt.plot(Temperature, abs(Magnetization), '*', color="#348ABD", label='Magnetization');
plt.errorbar(Temperature, abs(Magnetization), fmt='none', xerr=0, yerr= Error_Mag)
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Magnetization ", fontsize=20);


sp =  f.add_subplot(2, 2, 3 );
plt.plot(Temperature, SpecificHeat, 'd', color='black', label='Specific Heat');
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Specific Heat ", fontsize=20);


sp =  f.add_subplot(2, 2, 4 );
plt.plot(Temperature, Susceptibility, 's', label='Specific Heat');
plt.legend(loc='best', fontsize=15); 
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Suseptibility", fontsize=20);


plt.show()

'''
if sc_setting == "FiniteSize":
    f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
    for i in [8,16]:
        N, nt       = i, 10
        eqSteps, mcSteps = 1000, 1000
    
        Field           = np.zeros((N+2,N+2), dtype=np.int64)
        Energy          = np.zeros((nt), dtype=np.float64)
        Magnetization   = np.zeros((nt), dtype=np.float64)
        SpecificHeat    = np.zeros((nt), dtype=np.float64)
        Susceptibility  = np.zeros((nt), dtype=np.float64)
        Error_Ene       = np.zeros((nt), dtype=np.float64)
        Error_Mag       = np.zeros((nt), dtype=np.float64)
        Binder          = np.zeros((nt), dtype=np.float64)
        #Temperature
        Temperature = np.linspace(2,3, nt)
        Beta = 1.0/Temperature   # set k_B = 1
    
        #instantiate the class Ising model
    #        Ising = Ising(N, nt, eqSteps, mcSteps)
    
        twoD(Field, Energy, Magnetization, SpecificHeat, Susceptibility, Beta, Error_Ene, Error_Mag, Binder)
    
        #sp =  f.add_subplot(2, 2, (i+1) ); 
        ## Finite size scaling of suseptibility: chi(L) = chi*L^(-\gamma/nu)
#        plt.plot(Temperature, Susceptibility*N**(-7.0/4.0), label='N=%s'%N);
        plt.plot(Temperature, Binder, label='N=%s'%N)
        plt.legend(loc='best', fontsize=15); 
        plt.ylim(bottom = 1)
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Binder Ratio (Q)", fontsize=20);
    
    
    plt.show()
'''