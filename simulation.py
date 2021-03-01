#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:17:01 2021

@author: artemstopnevich
"""


import numpy as np
import math 
from datetime import datetime
import matplotlib.pyplot as plt
import ising

start = datetime.now()


#%%

print('Monte Carlo Simulation of 2D Ising Model using Metropolis Algorithm')
print("---------------------")

# Ising model: multiple monte-carlo sims or snapshots of configuration evolution @ set temp (simulation/configuration/critical)
sc_Ising = "simulation"   
print("model type: ", sc_Ising)
print("---------------------")

dims    = [8]   # lattice dimensions NxN (can be given as a list)
n       = 8             # configuration

nt      = 50           # number of temperature points
eqSteps = 1000           # number of MC iterations for equilibration
mcSteps = 10000         # number of MC iterations for calculation
minTemp = 1.0
maxTemp = 4.0

temp    = 1         # set temperature for Ising model evolution (between 0 and 7)
mcTime  = 1001      # number of time iterations for equilibration (coarsening)
    
print("Parameters:")
#%%
  
Temperatures        = dict()
Energies            = dict()     
Magnetizations      = dict()  
SpecificHeats       = dict()
Susceptibilities    = dict()
Mag_Densities       = dict()
Errors_E            = dict()
Errors_M            = dict()
Errors_C            = dict()

for N in dims:
    
    Spin            = np.zeros((N,N), dtype=np.int32)
    Energy          = np.zeros((nt), dtype=np.float64)
    Magnetization   = np.zeros((nt), dtype=np.float64)
    SpecificHeat    = np.zeros((nt), dtype=np.float64)
    Susceptibility  = np.zeros((nt), dtype=np.float64)
    Magnetization_T = np.zeros((nt), dtype=np.float64)
    SpecificHeat_T  = np.zeros((nt), dtype=np.float64)
    Mag_Density     = np.zeros((nt), dtype=np.float64)
    Error_E         = np.zeros((nt), dtype=np.float64)
    Error_M         = np.zeros((nt), dtype=np.float64)
    Error_C         = np.zeros((nt), dtype=np.float64)
    
    Temperature = np.linspace(minTemp, maxTemp, nt)
    Beta = 1/Temperature            # set k_B = 1

    #call class Ising 
    Ising = ising.Ising(N, nt, eqSteps, mcSteps)
    #Ising 2D 
    if sc_Ising == "simulation":
        print("N: ", N, "nt:", nt, "eqSteps:", eqSteps, "mcSteps:", mcSteps)
        print("---------------------")
        Energy, Magnetization, SpecificHeat, Susceptibility, SpecificHeat_T, Magnetization_T, Error_E, Error_M, Error_C = Ising.configData(Spin, Energy, Magnetization, SpecificHeat, Susceptibility, Beta, SpecificHeat_T, Magnetization_T, Error_E, Error_M, Error_C)
        
        Temperatures["N = %d"%N]        = Temperature 
        Energies["N = %d"%N]            = Energy    
        Magnetizations["N = %d"%N]      = Magnetization 
        Mag_Densities["N = %d"%N]       = Mag_Density
        SpecificHeats["N = %d"%N]       = SpecificHeat
        Susceptibilities["N = %d"%N]    = Susceptibility
        Errors_E["N = %d"%N]            = Error_E
        Errors_M["N = %d"%N]            = Error_M
        Errors_C["N = %d"%N]            = Error_C
'''
    if sc_Ising == "configuration":
        print("N:", n, "temp:", temp, "mcTime:", mcTime)
        print("---------------------")
        Ising.simulate(n, temp, mcTime)
    
    if sc_Ising == "critical":
        print("N: ", dims, "nt:", nt, "eqSteps:", eqSteps, "mcSteps:", mcSteps, "minTemp:", minTemp, "maxTemp:", maxTemp)
        print("---------------------")
        lattices, data = Ising.CriticalTemp(dims, nt, eqSteps, mcSteps)
        Ising.plotTemp_C(lattices, nt, data)     
'''
#%%
    
# Visualization
if sc_Ising == "simulation":
    f = plt.figure(figsize=(18, 10), dpi=80, linewidth=3, edgecolor='k')   
    f.suptitle("Monte Carlo simulation of the Ising model", fontsize = 20)
    
    sp =  f.add_subplot(2, 2, 1 )
    for dim in Energies.keys():
        plt.plot(Temperature, Energies[dim], 'o', label='{}'.format(dim))
        plt.errorbar(Temperature, Energies[dim], fmt='none', xerr=0, yerr= Errors_E[dim])
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy ", fontsize=20)
    plt.legend()
    
    sp =  f.add_subplot(2, 2, 2 )
    for dim in Magnetizations.keys():
        plt.plot(Temperature, abs(Magnetizations[dim]), '*', label='{}'.format(dim))
        plt.errorbar(Temperature, abs(Magnetizations[dim]), fmt='none', xerr=0, yerr=Errors_M[dim])
    plt.plot(Temperature, abs(Magnetization_T), color='black', label='Exact solution')
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.legend()
    
    
    sp =  f.add_subplot(2, 2, 3 )
    for dim in SpecificHeats.keys():
        plt.plot(Temperature, SpecificHeats[dim], 'd', label='{}'.format(dim))
        plt.errorbar(Temperature, SpecificHeats[dim], fmt='none', xerr=0, yerr=Errors_C[dim])
    plt.plot(Temperature, SpecificHeat_T, color='black', label='Exact solution')
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat ", fontsize=20)
    plt.legend()
    
    
    sp =  f.add_subplot(2, 2, 4 )
    for dim in Susceptibilities.keys():
        plt.plot(Temperature, Susceptibilities[dim], 's', label='{}'.format(dim))
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.legend()
    
    f.savefig("Results/Data_{}.png".format(dims))
    plt.show()


end = datetime.now() - start
print("---------------------")
print("Execution time: ", end)

