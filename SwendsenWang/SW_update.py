#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:57:22 2021

@author: artemstopnevich
"""


import numpy as np
import pandas as pd
from numpy import transpose as tr
from copy import copy
import matplotlib.pyplot as plt
from core_functions.physical_measures import *
from core_functions.nearest_neighbours import *
from core_functions.blocking_method import *




def add_NN(Lattice, bonded, cluster, K, N, s_i):
    '''
    check and add nearest neighbors to a cluster
    :param Lattice:
    :param bonded: visited or not?
    :param clusters: list of elements belonging to cluster
    :param K: coupling constant strength (beta*J)
    :param N: size of lattice
    :param s_i: current site or root site
    :return:
    '''
    new_cluster_sites = []
    p = 1-np.exp(-2*K)
    num_NN = 1
    NN = getNN(s_i, N, num_NN);
    bonded[tuple(s_i)] = -1;
    for NN_coords in NN:
        # accepting condition:
        NN_spin = Lattice[tuple(NN_coords)]
        if(NN_spin == Lattice[tuple(s_i)]):
            r = np.random.rand();
            if( r < p and bonded[tuple(NN_coords)] == 1):
                new_cluster_sites.append(tuple(NN_coords))   # mark it as new member of cluster
                cluster.append(tuple(NN_coords))     # add to cluster
                bonded[tuple(NN_coords)] = -1;
    return new_cluster_sites, bonded


## we need an efficient way of generating the clusters
def generate_cluster(Lattice, visited, K, root, search_cutoff = 500):
    '''
    this is the graph search like function
    :param root: coordinate to generate cluster from
    :param Lattice: ising grid
    :param K: J*beta
    :return: coordinates of a single new cluster in as a list of tuples
    '''
    N = tuple(Lattice.shape);
    bonded = np.ones(N)
    #start = (0, 0)
    if(visited[root] == 0):
        queue = [root];
        cluster = [root];
        visited[root] = 1;c = 0;
        while (len(queue) > 0): ## cluster formation can become expensive when the clusters get large on large grid
            s_i = queue.pop(0);
            new_sites, bonded = add_NN(Lattice, bonded, cluster, K, N, s_i);
            for site in new_sites:
                queue.append(site)
            c+=1;
            if(c > search_cutoff): #safety measure for computation 
                break;
        return cluster, bonded;
    else:
        return [], bonded;


## open question, how do we search the grid to search for viable clusters?
def run_cluster_sim(Lattice, epochs, N, J, beta, 
                    disp_cutoff = 100, visual = False, error=True):
    ''' run cluster epoch simulations '''
    relax_time =100;
#    Lattice = 2 * np.random.randint(0, 2, N) - 1; #put random lattice at the beginning
    bond_history = np.zeros(N);
    M = E = M2 = E2 = 0
    data = list();
    K = J*beta;
    for t in range(epochs):
        #select site to start cluster
        root = [];
        for i in range(len(N)):
            root.append(np.random.randint(0,N[i],1)[0]) #choosing random site
        root = tuple(root);
        visited = np.zeros(N);
        cluster, bonded = generate_cluster(Lattice, visited, K, root)
        bond_history += bonded;
        Lattice = Lattice*bonded;
        if(t > relax_time):
            M = abs(magnetization(Lattice));
            E = energy(Lattice);
            M2 = M*M;
            E2 = E*E;
            data.append([M, E, M2, E2]);

        if visual == True and (t%disp_cutoff ==0):
#            print('epoch: '+str(t))
            plt.suptitle("Epochs: {}".format(t));
            plt.subplot('121');
            plt.imshow(Lattice[:,:]);
#            plt.colorbar();
            plt.title("Lattice");
            plt.subplot('122');
            plt.imshow(bond_history[:,:]);
            bond_history = np.zeros(N);
            plt.colorbar();
            plt.draw();
            plt.title("Bonds");
            plt.show();
    data = np.array(data);
    
    if error == True:

        error_E = blocking_error(data[:,0], 10);
        error_M = blocking_error(data[:,0], 10);
        errors = np.array([error_E, error_M]);
    else:
        errors = []
    data = np.mean(data, axis = 0);
    data = np.append(data, [susceptibility(data[0], data[2], beta)], axis=0);
    data = np.append(data, [heat_capacity(data[1], data[3], beta)], axis=0);
    data = np.append(data, [binder(data[0], data[2])], axis=0);
    return data, errors

def run_clusters(Lattice, epochs, N, J, T, nT):
    data = np.zeros((7,nT));
    errors = np.zeros((2,nT));
    for tt in range(nT):
        print(tt,nT);
        beta = 1/T[tt];
        K = J*beta;
        idata, ierrors = run_cluster_sim(Lattice, epochs, N, J, beta,  disp_cutoff=100, error=True);
        data[:,tt] = idata;
        errors[:,tt] = ierrors;
    df1 = pd.DataFrame(tr(data), columns = ["mag", "ene", "mag^2", "ene^2", "susc", "sp_heat", "binder"])
    df2 = pd.DataFrame(tr(errors), columns = ["mag", "ene"])
    return df1, df2

def run_binder(Lattice, epochs, Ls, J, T, nT):
    for L in Ls:
        N = (L,L);
        data, _ = run_clusters(epochs, N, J, T, nT);
        plt.plot(T, data.binder, label='N=%s'%L);
        plt.legend(loc='best', fontsize=15); 
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Binder Ratio (Q)", fontsize=20);
    plt.savefig("figures/Binder_{}".format(Ls));
    plt.show()