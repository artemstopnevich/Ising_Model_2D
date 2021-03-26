#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:03:42 2021

@author: artemstopnevich
"""

import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.SW_update import *


# specify dimensions to generate 2D grids
# Note to have fairly small difference between lattice sizes for better results
L = [8,16,32]
nT = 2; J = 1

#simulation parameters
epochs = 10000;
nearest_neighbors = 1;


Temperature = np.linspace(2.2, 2.3, nT);

run_binder(epochs, L, J, Temperature, nT)

