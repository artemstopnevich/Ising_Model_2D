#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:03:42 2021

@author: artemstopnevich
"""

import numpy as np
import matplotlib.pyplot as plt
from Wolff.wolff_algorithm import *


# Note to have fairly small difference between lattice sizes for better results

#simulation parameters
Ls = [10]
nT = 2; 
epochs = 2000;


Energy = np.zeros((nT), dtype = np.float64)
Magnetisation = np.zeros((nT), dtype = np.float64)
Binder = np.zeros((nT), dtype = np.float64)
Error_E = np.zeros((nT), dtype = np.float64)
Error_M = np.zeros((nT), dtype = np.float64)
Error_Q = np.zeros((nT), dtype = np.float64)

Temperature = np.linspace(2.23, 2.3, nT);
Beta = 1/Temperature

run_binder_wolff(Energy, Magnetisation, Binder, Error_E, Error_M, Error_Q, Beta, Temperature, epochs, Ls, nT)

