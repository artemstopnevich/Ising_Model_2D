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


# model parameters
N = 32
nT = 100
epochs = 8000

# observables arrays
Field           = np.zeros((N+2,N+2), dtype=np.int32)
Energy          = np.zeros((nT), dtype=np.float64)
Magnetization   = np.zeros((nT), dtype=np.float64)
SpecificHeat    = np.zeros((nT), dtype=np.float64)
Susceptibility  = np.zeros((nT), dtype=np.float64)
Binder          = np.zeros((nT), dtype=np.float64)
#Error_Ene       = np.zeros((nT), dtype=np.float64)
#Error_Mag       = np.zeros((nT), dtype=np.float64)

#Temperature
Temperature = np.linspace(1.0,4.0, nT)
Beta = 1.0/Temperature   # set k_B = 1

# model execution
Ising = ising.Ising(N, nT, epochs)

Ising.twoD(Field, Energy, Magnetization, SpecificHeat, Susceptibility, 
           Beta, Binder)



# Visualization
observables_fig_Metropolis(Temperature, Energy, Magnetization, SpecificHeat, Susceptibility)

