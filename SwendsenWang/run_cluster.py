#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created oN2 Mon Mar 22 21:49:34 2021

@author: artemstopnevich
"""


import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.SW_update import *
from core_functions.physical_measures import observables_fig




# initialise 2D grid
L = 32;
N = (L,L);

#simulation parameters
nT = 20; J = 1;
epochs = 2000;
nearest_neighbors = 1;

Temperature = np.linspace(1,4, nT);
Lattice = 2 * np.random.randint(0, 2, N) - 1;

data, errors = run_clusters(Lattice, epochs, N, J, Temperature, nT);

observables_fig(Temperature, data, errors, L, epochs) # plot observables


