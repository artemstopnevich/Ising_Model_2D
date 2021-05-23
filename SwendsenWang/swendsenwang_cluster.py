#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:27:39 2021

@author: artemstopnevich
"""

import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.swendsenwang_algorithm import *
from core_functions.physical_measures import observables_fig



#simulation parameters
Dims = [16]
epochs = 500;
nT = 50; 

f = plt.figure(figsize=(18, 6), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
f.suptitle("Observables with {} iterations".format(epochs), fontsize=20);
sp1 =  f.add_subplot(1, 2, 1 );
sp2 =  f.add_subplot(1, 2, 2 );
sp1.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
sp2.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
for L in Dims:
    N = (L,L);

    Lattice = np.zeros(N, dtype = np.int32)
    
    Energy = np.zeros((nT), dtype = np.float64)
    Magnetisation = np.zeros((nT), dtype = np.float64)
    #SpecificHeat = np.zeros((nT), dtype = np.float64)
    #Susceptibility = np.zeros((nT), dtype = np.float64)
    Binder = np.zeros((nT), dtype = np.float64)
    Error_E = np.zeros((nT), dtype = np.float64)
    Error_M = np.zeros((nT), dtype = np.float64)
    Error_Q = np.zeros((nT), dtype = np.float64)
    
    Temperature = np.linspace(1.0,4.0,nT)
    Beta = 1.0/Temperature
    
    run_clusters_sw(Lattice, Energy, Magnetisation, Binder, Error_E, Error_M, Beta, epochs, N, nT)


#%%
f = plt.figure(figsize=(18, 6), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
f.suptitle("Physical Observables (Swendsen-Wang algorithm)".format(epochs), fontsize=20);
sp1 =  f.add_subplot(1, 2, 1 );
sp2 =  f.add_subplot(1, 2, 2 );
sp1.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
sp2.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
observables_fig(Temperature, Energy, Magnetisation, Error_E/6, Error_M/6, 64, epochs,f,sp1,sp2, 's', 'k') # plot observables

plt.show()
#plt.savefig("figures/observables_{}_grid_{}_steps.png".format(L, epochs))