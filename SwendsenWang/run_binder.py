#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:03:42 2021

@author: artemstopnevich
"""

import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.SW_update import *


# initialise 2D grid
L = [10,15,20,25]
nT = 30; J = 1

#simulation parameters
epochs = 2000;
nearest_neighbors = 1;

Temperature = np.linspace(2.1, 2.4, nT)

data = run_binder(epochs, L, J, Temperature, nT)

