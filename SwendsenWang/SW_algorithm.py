#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:36:17 2021

@author: artemstopnevich
"""
import numpy as np
from core_functions.nearest_neighbours import *

#-----------------------------------------------------------------------
# Swendsen-Wang algorithm function
#-----------------------------------------------------------------------

def SW_BFS(lattice, bonded, clusters, root, beta, J, nearest_neighbors = 1):

    p = 1 - np.exp(-2 * beta * J); #bond forming probability
    N = lattice.shape;
    visited = np.zeros(N); #indexes whether we have visited nodes during
                                 #this particular BFS search
    queue = list();
    if(bonded[tuple(root)] != 0): #cannot construct a cluster from this site
        return bonded, clusters, visited;
    
    queue.append(root);
    s_i = lattice[tuple(root)];
    color = np.max(bonded) + 1;
    
    index = np.ravel_multi_index(tuple(root), dims=tuple(N), order='C')
    clusters[index] = list();

    while(len(queue) > 0):
        #print(queue)
        r = tuple(queue.pop(0));
        
        if(visited[r] == 0): #if not visited
            visited[r] = 1;
            clusters[index].append(r);

            bonded[r] = color;
            NN = getNN(r,N, nearest_neighbors);
            for nn_coords in NN:
                rn = tuple(nn_coords);
                if(lattice[rn] == s_i and bonded[rn] == 0 and visited[rn] == 0): 
                    #require spins to be aligned
                    if (np.random.rand() < p):  # accept bond proposal
                        queue.append(rn); #add coordinate to search
                        clusters[index].append(rn) 
                        bonded[rn] = color; #indicate site is no longer available
    return bonded, clusters, visited;


'''
# Simplest cluster algorithm
# params
N = 16
epochs = 1000

def cluster(N, epochs, nt, T, E, S):
    E1 = 0
    N2 = N * N

    S = np.random.choice([1, -1], N2) 
    nbr = {i : ((i // N) * N + (i + 1) % N, (i + N) % N2,
            (i // N) * N + (i - 1) % N, (i - N) % N2)
                                    for i in range(N2)}

    p = 1.0 - np.exp(-2.0 / T[tt])
    for step in range(nsteps):
        k = np.random.randint(0, N - 1)
        Pocket, Cluster = [k], [k]
        while Pocket != []:
            j = np.random.choice(Pocket)
            for l in nb[j]:
                if S[l] == S[j] and l not in Cluster \
                       and np.random.uniform(0.0, 1.0) < p:
                    Pocket.append(l)
                    Cluster.append(l)
            Pocket.remove(j)
        for j in Cluster:
            S[j] *= -1
    return S

Spins = cluster(N, nsteps, nt, Temperature, Energy, Spins)


'''