#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:16:57 2021

@author: artemstopnevich
"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import pyximport
pyximport.install()
import Metropolis.ising as ising

f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, facecolor='w', edgecolor='k');  
plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')

# specify dimensions to test
for i in [5,10,15]:
    
    N, nT       = i, 10
    epochs = 2000

    Field           = np.zeros((N+2,N+2), dtype=np.int32)
    Energy          = np.zeros((nT), dtype=np.float64)
    Magnetization   = np.zeros((nT), dtype=np.float64)
    SpecificHeat    = np.zeros((nT), dtype=np.float64)
    Susceptibility  = np.zeros((nT), dtype=np.float64)
    Error_Ene       = np.zeros((nT), dtype=np.float64)
    Error_Mag       = np.zeros((nT), dtype=np.float64)
    Binder          = np.zeros((nT), dtype=np.float64)
    
    #Temperature
    Temperature = np.linspace(2,3, nT)
    Beta = 1.0/Temperature   # set k_B = 1

    #instantiate the class Ising model
    Ising = ising.Ising(N, nT, epochs)
    
    Ising.twoD(Field, Energy, Magnetization, SpecificHeat, Susceptibility, 
               Beta, Binder, Error_Ene, Error_Mag)


    plt.plot(Temperature, Binder, label='N=%s'%N)
    plt.legend(loc='best', fontsize=15); 
    plt.ylim(bottom = 1)
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Binder Ratio (Q)", fontsize=20);


plt.show()

#sp =  f.add_subplot(2, 2, (i+1) ); 
## Finite size scaling of suseptibility: chi(L) = chi*L^(-\gamma/nu)
#  plt.plot(Temperature, Susceptibility*N**(-7.0/4.0), label='N=%s'%N);