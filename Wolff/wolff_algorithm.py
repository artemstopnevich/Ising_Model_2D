#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:57:22 2021

@author: artemstopnevich
"""


import numpy as np
import pandas as pd
import math
from numpy import transpose as tr
from copy import copy
import matplotlib.pyplot as plt
from core_functions.physical_measures import *
from core_functions.nearest_neighbours import *
from core_functions.blocking_method import *

#-----------------------------------------------------------------------
# Wolff algorithm 
#-----------------------------------------------------------------------
def run_cluster_wolff(Lattice, E, M, Q, EE, EM, EQ, B, epochs, N, nT, relax_time=100):

    initialise(Lattice, N[0])
    for tt in range(nT):
        print(tt,nT);
        beta = B[tt];
#        N = Lattice.shape;
        iMCs = 1.0/(epochs-relax_time);
        iNs = 1.0/np.prod(N);
        M1 = E1 = M2 = E2 = 0
        Mag = 0
        Ene = 0
        data_E = np.zeros((epochs), dtype = np.float64);
        data_M = np.zeros((epochs), dtype = np.float64);
        data_M2 = np.zeros((epochs), dtype = np.float64);
        # generate random particle

        p = 1 - np.exp(-2 * beta)
        for t in range(epochs):
            bonded = np.ones(N);
            visited = np.zeros(N);
            root = []; # generate random coordinate by sampling from uniform random...
            queue = list();
            for i in range(len(N)):
                root.append(np.random.randint(0, N[i], 1)[0])
            root = tuple(root);
            visited[root]=1;
            Cluster = [root];  # denotes cluster coordinates
            queue.append(root);  # old frontier
            bonded[root] = -1;
            while (len(queue) != 0):
                site = tuple(queue.pop(0))
    #            for site in F_old:
                site_spin = Lattice[tuple(site)]
                # get neighbors
                NN_list = getNN(site, N, num_NN=1);
                for NN_site in NN_list: ## if we do the full search, this is bad, because
                    nn = tuple(NN_site)
                    if (Lattice[nn] == site_spin and visited[nn] == 0):
                        if (np.random.rand() < p):
                            queue.append(nn); visited[nn] = 1;
                            Cluster.append(nn);
                            bonded[nn] = -1;

            Lattice = Lattice*bonded;
            # update the cluster

            if(t > relax_time):
                Ene = calcEnergy(Lattice);                 
                Mag = calcMag(Lattice);
                E1 = E1 + Ene;
                M1 = M1 + Mag;
                E2 = E2 + Ene*Ene;
                M2 = M2 + Mag*Mag;
                for site in Cluster:
                    Lattice[site] = -1 * Lattice[site]  
                    
                error_e = blocking_method(Ene, t, 100, epochs, iNs, data_E)
                error_m = blocking_method(Mag, t, 100, epochs, iNs, data_M)
                error_m2 = blocking_method(Mag*Mag, t, 100, epochs, iNs, data_M2)
            
        E[tt] = E1*iMCs*iNs 
        M[tt] = M1*iMCs*iNs 
#        C[tt] = (E2*iMCs - E1*E1*iMCs*iMCs)*beta*beta*iNs;
#        X[tt] = (M2*iMCs - M1*M1*iMCs*iMCs)*beta*iNs;
        Q[tt] = (M2*iMCs)/(M1*M1*iMCs*iMCs)   
        EE[tt] = error_e
        EM[tt] = error_m
        EQ[tt] = error_m2/(2*abs(M1)*error_m)
        
    return  

def run_binder_wolff(E, M, Q, EE, EM, EQ, B, T, epochs, Ls, nT):
    for L in Ls:
        N = (L,L);
#        Lattice = 2 * np.random.randint(0, 2, N) - 1;
        Lattice = np.zeros(N, dtype = np.int32)        
        run_cluster_wolff(Lattice, E, M, Q, EE, EM, EQ, B, epochs, N, nT)
        f = plt.figure(dpi=180)
        plt.plot(T, Q, label='N=%s'%L);
        plt.errorbar(T, Q, fmt='none', xerr=0, yerr= EQ);
        plt.legend(loc='best', fontsize=15); 
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Binder Ratio (Q)", fontsize=20);
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.ylim(1.05,1.15);
    plt.savefig("figures/Binder_{}".format(Ls));
    plt.show()
  
    
def initialise(Lattice, N):    # generates a random spin Spin configuration
    '''generates a random spin configuration for initial condition'''
    for i in range(N):
        for j in range(N):
            if(np.random.rand() < 0.5): 
                Lattice[i, j]=-1
            else:
                Lattice[i, j]=1
