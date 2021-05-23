#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:26:55 2021

@author: artemstopnevich
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from core_functions.physical_measures import *
import pyximport
pyximport.install()
import Metropolis.ising as ising
from datetime import datetime

start = datetime.now()

# model parameters
Dims = [32]
nT = 70
epochs = 5000
relax_time = 4000
f = plt.figure(figsize=(18, 6), dpi=160, linewidth=3, facecolor='w', edgecolor='k');   
f.suptitle('Physical Observables of Ising Model', fontsize=20) 
sp1 =  f.add_subplot(1, 2, 1);
sp2 =  f.add_subplot(1, 2, 2);
#sp3 =  f.add_subplot(2, 2, 3);
#sp4 =  f.add_subplot(2, 2, 4);
sp1.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
sp2.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
#sp3.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
#sp4.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r', label = 'T_crit')
colors = ['k', 'b', 'r']
markers = ['s', '*', 'o']
for N in Dims:
    # observables arrays
    Field           = np.zeros((N+2,N+2), dtype=np.int32)
    Energy          = np.zeros((nT), dtype=np.float64)
    Magnetization   = np.zeros((nT), dtype=np.float64)
    SpecificHeat          = np.zeros((nT), dtype=np.float64)
    Susceptibility   = np.zeros((nT), dtype=np.float64)    
    Binder          = np.zeros((nT), dtype=np.float64)
    Error_Ene       = np.zeros((nT), dtype=np.float64)
    Error_Mag       = np.zeros((nT), dtype=np.float64)
    Error_C       = np.zeros((nT), dtype=np.float64)
    Error_X       = np.zeros((nT), dtype=np.float64)
    #Temperature
    Temperature = np.linspace(0,4, nT)
    Beta = 1.0/Temperature   # set k_B = 1
    
    # model execution
    Ising = ising.Ising(N, nT, epochs, relax_time)
    
    Ising.twoD(Field, Energy, Magnetization,SpecificHeat, Susceptibility,
               Beta, Binder, Error_Ene, Error_Mag, Error_C, Error_X)
    
    # Visualization
    c = colors[Dims.index(N)]
    m = markers[Dims.index(N)]
    Error_C       = np.zeros((nT), dtype=np.float64)
    Error_X       = np.zeros((nT), dtype=np.float64)
    observables_fig_Metropolis(Temperature, Energy, Magnetization, SpecificHeat, Susceptibility, Error_Ene, Error_Mag, Error_C, Error_X, N, f, sp1, sp2, c,m)

plt.show()

end = datetime.now() - start
print(end)