#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:27:39 2021

@author: artemstopnevich
"""

import numpy as np
import matplotlib.pyplot as plt
from SwendsenWang.SW_algorithm import *
## basic tests of cluster generation
#
L = 20
N = (L,L)
lattice = 2*np.random.randint(0,2,N)-1

clusters = dict();
plt.imshow(lattice[:,:]);

bonded = np.zeros(N);

beta = 2; J = 1;
p = 1-np.exp(-2*J*beta);


i = 0; j = 0;
bonded, clusters, visited = SW_BFS(lattice, bonded, clusters, [i, j], beta, J);

print(len(clusters))
print(bonded)
xr = list(range(0,L));
yr = list(range(0,L));
[Xr, Yr] = np.meshgrid(xr,yr)
plt.figure()
plt.scatter(Xr.flatten(), Yr.flatten(), c = bonded[:,:].ravel(), cmap ='jet');

plt.show()


