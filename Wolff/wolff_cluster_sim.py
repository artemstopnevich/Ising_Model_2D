#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:32:32 2021

@author: artemstopnevich
"""


import numpy as np
import matplotlib.pyplot as plt
from Wolff.wolff_algorithm import *
from core_functions.physical_measures import observables_fig



# Settings
#sc_visual = False

# initialise 2D grid
L = 64
N = (L,L)
#Lattice = np.random.choice([-1,1], N)
T = 2.2;
beta = 1.0/T; 

#simulation parameters
epochs = 2000;
nearest_neighbors = 1;

Lattice = 2 * np.random.randint(0, 2, N) - 1;

run_cluster_sim_wolff(Lattice, epochs, N, beta, disp_cutoff = 100)