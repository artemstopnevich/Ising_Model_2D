#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:36:17 2021

@author: artemstopnevich
"""
import numpy as np
from core_functions.nearest_neighbours import *
from core_functions.physical_measures import *
from core_functions.blocking_method import *

#-----------------------------------------------------------------------
# Swendsen-Wang algorithm 
#-----------------------------------------------------------------------

def SW_BFS(lattice, bonded, clusters, root, beta, nearest_neighbors = 1):

    p = 1 - np.exp(-2 * beta); 
    N = lattice.shape;
    visited = np.zeros(N); 
    queue = list();
    if(bonded[tuple(root)] != 0): 
        return bonded, clusters, visited;
    
    queue.append(root);
    s_i = lattice[tuple(root)];
    color = np.max(bonded) + 1;
    index = np.ravel_multi_index(tuple(root), dims=tuple(N), order='C')
    clusters[index] = list();

    while(len(queue) > 0):
        r = tuple(queue.pop(0));     
        if(visited[r] == 0): 
            visited[r] = 1;
            clusters[index].append(r);
            bonded[r] = color;
            NN = getNN(r,N, nearest_neighbors);
            for nn_coords in NN:
                rn = tuple(nn_coords);
                if(lattice[rn] == s_i and bonded[rn] == 0 and visited[rn] == 0): 
                    #require spins to be aligned
                    if (np.random.rand() < p):  
                        queue.append(rn); 
                        clusters[index].append(rn) 
                        bonded[rn] = color;  #available
    return bonded, clusters, visited;



def generate_clusters(Lattice, epochs, beta, N):
    
    relax_time = 100
    iMCs = 1.0/(epochs-relax_time);
    iNs = 1.0/np.prod(N);
    M1 = M2 = E1 = E2 = 0; 
    Mag = 0
    Ene = 0
    data_E = np.zeros((epochs), dtype = np.float64);
    data_M = np.zeros((epochs), dtype = np.float64);
    data_M2 = np.zeros((epochs), dtype = np.float64);
#    M = np.zeros((epochs), dtype = np.float64)
    for t in range(epochs):
        bonded = np.zeros(N);
        clusters = dict();  #
        for ii in range(np.prod(N)):
            a = np.random.randint(0,N[0])
            b = np.random.randint(0,N[1])
#        for i in range(N[0]):
#            for j in range(N[1]):
            bonded, clusters, visited = SW_BFS(Lattice, bonded, clusters, [a,b], beta);
        for cluster_index in clusters.keys():
            r = np.unravel_index(cluster_index, N);
            p = np.random.rand();
            if(p < 0.5):
                for coords in clusters[cluster_index]:
                    Lattice[tuple(coords)] = -1*Lattice[tuple(coords)];               
        if (t > relax_time):
#            M[t] = abs(calcMag(Lattice))
            Mag = calcMag(Lattice)
            Ene = calcEnergy(Lattice)
            E1 = E1 + Ene;
            M1 = M1 + Mag;
            E2 = E2 + Ene*Ene;
            M2 = M2 + Mag*Mag;
            
            error_e = blocking_method(Ene, t, 10, epochs, iNs, data_E)
            error_m = blocking_method(Mag, t, 10, epochs, iNs, data_M)
            error_m2 = blocking_method(Mag*Mag, t, 10, epochs, iNs, data_M2) 
            
    E = E1*iMCs*iNs
    M = M1*iMCs*iNs
#    C = (E2*iMCs - E1*E1*iMCs*iMCs*iNs)*iNs*beta*beta; # sort out
#    X = (M2*iMCs - abs(M1)*abs(M1)*iMCs*iMCs*iNs)*beta*iNs;
    Q = (M2*iMCs)/(M1*M1*iMCs*iMCs)    
    EE = error_e
    EM = error_m
    return E, M, Q, EE, EM
#    return M
            
            
        
def run_clusters_sw(L, E, M, Q, EE, EM, B, epochs, N, nT):
    initialise(L, N[0])
    for tt in range(nT):
        print(tt,nT);
        beta = B[tt];
        E[tt], M[tt], Q[tt], EE[tt], EM[tt] = generate_clusters(L, epochs, beta, N)
    return 
        
def initialise(Lattice, N):    # generates a random spin configuration
    '''generates a random spin configuration for initial condition'''
    for i in range(N):
        for j in range(N):
            if(np.random.rand() < 0.5): 
                Lattice[i, j]=-1
            else:
                Lattice[i, j]=1    