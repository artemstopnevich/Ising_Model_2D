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


def add_NN(Lattice, bonded, cluster, K, N, s_i):
    '''
    check and add nearest neighbors to a cluster
    
    Parameters
    ----------
    Lattice : Randomly configured lattice of spins [-1,1].
    
    bonded: dummy array of ones
    
    clusters: empty dictionary, which will store all the clusters

    K : J*beta
    
    N : dimensions of the lattice (tuple).
    
    s_i: current site or root site
    
    Returns
    -------
    new_cluster_sites: newly identified sites, which belong to the
                        cluster of root site

    bonded: 1 or 0, indicates whether a site has been assigned to 
             a cluster or not
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
    
    Parameters
    ----------
    Lattice : Randomly configured lattice of spins [-1,1].
    
    visited : array that tells, which sites have/ haven't already 
            been visited 
    K : J*beta
    
    root : coordinate to generate cluster from
    
    search_cutoff : stops searching for new cluster members.
                    The default is 500.
                    
    Returns
    -------
    cluster: identified cluster with coordiantes of sites as tuples

    bonded: 1 or 0, indicates whether a site has been assigned to 
                a cluster or not
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


## one to answer: how do we search the lattice to search for viable clusters? :)
def run_cluster_sim(Lattice, epochs, N, J, beta, 
                    disp_cutoff = 100, visual = False):
    '''
    Parameters
    ----------
    Lattice : randomly configurated lattice of spins/ 
              last config from previous temp. point.
    epochs : number of sweeps through cluster search.
    
    N : dimensions of the lattice (tuple).
    
    J : coupling coefficient (usually set to 1).
    
    beta : 1/Temperature
    
    disp_cutoff : divider of epoch sims. The default is 100.
    
    visual : Bool, visualise epoch sims. The default is False.
    
    error : Bool, choosing to calculate error for observables.
            The default is True.

    Returns
    -------
    data: physical observables
        
    errors: errors of physical observables
    '''

    relax_time =300;
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
            M = magnetization(Lattice);
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
    

    error_E = blocking_error(data[:,1], 10);
    error_M = blocking_error(data[:,0], 10);


    data = np.mean(data, axis = 0);
    data = np.append(data, [susceptibility(data[0], data[2], beta)]);
    data = np.append(data, [heat_capacity(data[1], data[3], beta)]);
    data = np.append(data, [binder(data[0], data[2])]);
    
    error_Q = (data[2]/data[0])*np.sqrt(((2*data[0]*error_M)/data[2])**2 + (error_M/data[0])**2)
    
    errors = np.array([error_E, error_M, error_Q]);
    return data, errors

def run_clusters(Lattice, epochs, N, J, T, nT):
    '''
    
    Parameters
    ----------
    Lattice : Randomly configured lattice of spins [-1,1].
    
    epochs : number of sweeps through cluster search.
    
    N : dimensions of the lattice (tuple).
    
    J : coupling coefficient (usually set to 1).
    
    T : temperature range 
    
    nT : number of temperature points

    Returns
    -------
    df1 : DataFrame of physical observables
    
    df2 : DataFrame of errors of observables

    '''
    data = np.zeros((7,nT));
    errors = np.zeros((3,nT));
    for tt in range(nT):
        print(tt,nT);
        beta = 1/T[tt];
        K = J*beta;
        idata, ierrors = run_cluster_sim(Lattice, epochs, N, J, beta,  disp_cutoff=100);
        data[:,tt] = idata;
        errors[:,tt] = ierrors;
    df1 = pd.DataFrame(tr(data), columns = ["mag", "ene", "mag^2", "ene^2", "susc", "sp_heat", "binder"])
    df2 = pd.DataFrame(tr(errors), columns = ["mag", "ene", "binder"])
    return df1, df2

def run_binder(epochs, Ls, J, T, nT):
    '''
    
    Parameters
    ----------
    Lattice : Randomly configured lattice of spins [-1,1].
    
    epochs : number of sweeps through cluster search.
    
    Ls : list of lattice sizes (lengths)
    
    J : coupling coefficient (usually set to 1).
    
    T : temperature range 
    
    nT : number of temperature points

    Returns
    -------
    Plotting the Binder Ratios to determine the critical temperature

    '''
    for L in Ls:
        N = (L,L);
        Lattice = 2 * np.random.randint(0, 2, N) - 1;
        data, errors = run_clusters(Lattice, epochs, N, J, T, nT);
        plt.plot(T, data.binder, label='N=%s'%L);

        plt.errorbar(T, data.binder, fmt='none', xerr=0, yerr= errors.binder);
        plt.legend(loc='best', fontsize=15); 
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Binder Ratio (Q)", fontsize=20);
    plt.axvline(x=2/math.log(1 + math.sqrt(2)), linestyle='--', color='r')
    plt.savefig("figures/Binder_{}".format(Ls));
    plt.show()