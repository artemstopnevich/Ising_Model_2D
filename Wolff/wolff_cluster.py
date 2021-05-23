#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created oN2 Mon Mar 22 21:49:34 2021

@author: artemstopnevich
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from Wolff.wolff_algorithm import *
from core_functions.physical_measures import observables_fig
#%%
#simulation parameters

Dims = [64]
epochs = 2000;
nT = 50; 

f = plt.figure(figsize=(18, 6), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
f.suptitle("Physical Observables (Wolff Algorithm)".format(epochs), fontsize=20);
sp1 =  f.add_subplot(1, 2, 1 );
sp2 =  f.add_subplot(1, 2, 2 );
sp1.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
sp2.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
colors = ['g', 'b', 'r']
markers = ['s', '*', 'o']
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
    
    run_cluster_wolff(Lattice, Energy, Magnetisation, Binder, Error_E, Error_M, Beta, epochs, N, nT);
    
    # Visualization
#%%
f = plt.figure(figsize=(18, 6), dpi=80, linewidth=3, facecolor='w', edgecolor='k');    
f.suptitle("Physical Observables (Wolff Algorithm)".format(epochs), fontsize=20);
sp1 =  f.add_subplot(1, 2, 1 );
sp2 =  f.add_subplot(1, 2, 2 );
sp1.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
sp2.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
c = colors[Dims.index(L)]
m = markers[Dims.index(L)]
observables_fig(Temperature, Energy, Magnetisation, Error_E/2, Error_M/1.5, L, epochs,f,sp1,sp2, m, 'k') #

plt.show()
plt.savefig("figures/observables_{}_grid_{}_steps.png".format(L, epochs))

