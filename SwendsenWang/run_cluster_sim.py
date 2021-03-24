#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:32:32 2021

@author: artemstopnevich
"""


import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.SW_update import *
from core_functions.physical_measures import observables_fig



# Settings
#sc_visual = False

# initialise 2D grid
L = 32
N = (L,L)
#Lattice = np.random.choice([-1,1], N)
T = 1.0
beta = 1.0/T; J = 1.0; 

#simulation parameters
epochs = 2000;
nearest_neighbors = 1;

run_cluster_sim(epochs, N, J, beta, disp_cutoff = 100, error=False)