#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:36:17 2021

@author: artemstopnevich
"""
import numpy as np
from core_functions.nearest_neighbours import *
# -----------------------------------------------------------------------------------------------------------------------
# Swendsen-Wang algorithm functions
# -----------------------------------------------------------------------------------------------------------------------

## this should be dimension independent
def SW_BFS(lattice, bonded, clusters, root, beta, J, nearest_neighbors = 1):
    '''
    function currently cannot generalize to dimensions higher than 2...
    main idea is that we populate a lattice with clusters according to SW using a BFS from a root coord
    :param lattice: lattice
    :param bonded: 1 or 0, indicates whether a site has been assigned to a cluster
           or not
    :param clusters: dictionary containing all existing clusters, keys are an integer
            denoting natural index of root of cluster
    :param start: root node of graph (x,y)
    :param beta: temperature
    :param J: strength of lattice coupling
    :param nearest_neighbors: number or NN to probe
    :return:
    '''
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

    ## need to make sub2ind work in arbitrary dimensions
    index = np.ravel_multi_index(tuple(root), dims=tuple(N), order='C')
    clusters[index] = list();
    #whatever the input coordinates are
    while(len(queue) > 0):
        #print(queue)
        r = tuple(queue.pop(0));
        ##print(x,y)
        if(visited[r] == 0): #if not visited
            visited[r] = 1;
            clusters[index].append(r);
            #to see clusters, always use different numbers
            bonded[r] = color;
            NN = getNN(r,N, nearest_neighbors);
            for nn_coords in NN:
                rn = tuple(nn_coords);
                if(lattice[rn] == s_i and bonded[rn] == 0 and visited[rn] == 0): 
                    #require spins to be aligned
                    if (np.random.rand() < p):  # accept bond proposal
                        queue.append(rn); #add coordinate to search
                        clusters[index].append(rn) #add point to the cluster
                        bonded[rn] = color; #indicate site is no longer available
    return bonded, clusters, visited;


'''
import numpy as np
import matplotlib.pyplot as plt

# constants
N = 16
nt = 20
nsteps = 1000

Spins           = np.zeros((N*N), dtype=np.int64)
Energy          = np.zeros((nt), dtype=np.float64)
Temperature     = np.linspace(1,4, nt)

# Main func
def cluster(N, nsteps, nt, T, E, S):
    E1 = 0
    N2 = N * N
    iMC = 1/nsteps
    iNs = 1/N
    S = initialise(S, N2)
    nb = neighbours(N, N2)

    for tt in range(nt):
        print(tt,nt)
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
            Ene = calcEnergy(S, N2, nb)
            E1 = E1 + Ene
        E[tt] = E1*iMC*iNs
    return S, E


# Complimentary funcs
def initialise(S, N2):
    S = np.random.choice([1, -1], N2) 
    return S

def neighbours(N, N2):
    nbr = {i : ((i // N) * N + (i + 1) % N, (i + N) % N2,
            (i // N) * N + (i - 1) % N, (i - N) % N2)
                                    for i in range(N2)}
    return nbr


def calcEnergy(S, N2, nb):
    energy = 0
    for i in range(N2):
        nb3 = nb[i][2]
        nb4 = nb[i][3]
        energy += -S[i]*(S[nb3] + S[nb4])
    return energy

Spins, Energy = cluster(N, nsteps, nt, Temperature, Energy, Spins)


plt.plot(Temperature, Energy)

'''