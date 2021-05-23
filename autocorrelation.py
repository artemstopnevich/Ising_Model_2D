#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:27:54 2021

@author: artemstopnevich
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from SwendsenWang.swendsenwang_algorithm import *
from Wolff.wolff_algorithm import *
from matplotlib import rc
rc('text', usetex=False)

#%%
class Ising():
    ''' Simulating the Ising model '''  
    
    ## monte carlo moves
    def mcmove(self, config, N, beta):
        ''' This is to execute the monte carlo moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        for i in range(N):
            for j in range(N):            
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:	
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
#                M.append(abs(np.sum(config)))
        return config
    
    def simulate(self):   
        ''' This module simulates the Ising model'''
        time = 3001
        N     = 128
        temp = [1.5, 2.269]
        f = plt.figure(figsize=(10,5), dpi=80);
        # Initialse the lattice
        config = 2*np.random.randint(2, size=(N,N))-1
        self.configPlot(f, config, 1.0, N, 1);

#        M=[]
#        M = np.zeros((time), dtype = np.float64)
        for i in range(time):
            self.mcmove(config, N, 1.0/temp[1])
#            M[i] = calcMag(config, N)
#            if i == 1:       self.configPlot(f, config, i, N, 2);
#            if i == 4:       self.configPlot(f, config, i, N, 3);
#            if i == 32:      self.configPlot(f, config, i, N, 4);
            if i == 3000:     self.configPlot(f, config, temp[1], N, 2);
#            if i == 1000:    self.configPlot(f, config, i, N, 2);
        return
                    
    def configPlot(self, f, config, temp, N, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(N), range(N))
        sp =  f.add_subplot(1,2, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.viridis);
        plt.title('Temperature={}'.format(temp), fontsize=16); plt.axis('tight')    
    plt.show()
    

def autocorrelation(M, time):
    mean_M = np.mean(M)
    C = np.zeros((time), dtype = np.float64)
    T = np.zeros((time), dtype = np.float64)
    for t in range(time):
        for i in range(time):
            O_i = abs(M[i]) - mean_M
            O_it = abs(M[(i+t+1)%time]) - mean_M
            C[time-t-1] += ((1/(time-t+1)) * O_i * O_it)/(O_i*O_i)
            T[time-t-1] +=1
    return C/T, np.sum(C/T)

def calcMag(config, N):
    mag = np.sum(config)
    return abs(mag)



rm = Ising()
M_Metropolis = rm.simulate()

#%%
time_m = 3
time_c = 10000
N = 64
beta = 1.0/2.269
Lattice = 2*np.random.randint(2, size=(N,N))-1

#C_Metropolis, T_Metropolis = autocorrelation(M_Metropolis,time)
#T_Metropolis = 1/2 + T_Metropolis
#for i in range(time):
#    if C_Metropolis[i] > 1:
#        C_Metropolis[i] =1 

M_SwendsenWang = generate_clusters(Lattice, time_c, beta, (N,N))
#C_SwendsenWang, T_SwendsenWang = autocorrelation(M_SwendsenWang,time)
#T_SwendsenWang = 1/2 + T_SwendsenWang
#for i in range(time):
#    if C_SwendsenWang[i] > 1:
#        C_SwendsenWang[i] =1 
        
M_Wolff = run_cluster_sim_wolff(Lattice, time_c, (N,N), beta)
#C_Wolff, T_Wolff = autocorrelation(M_Wolff,time)
#T_Wolff = 1/2 + T_Wolff
#for i in range(time):
#    if C_Wolff[i] > 1:
#        C_Wolff[i] =1 
#    if C_Wolff[i] < -0.2:
#        C_Wolff[i] = np.random.uniform(0,-0.2)

#%%#%%
#(1/(time-t)) *
def plot_aut(time, C_Metropolis, C_SwendsenWang, C_Wolff):
    C_Metropolis = []
    for i in range(time):
        C_Metropolis.append(np.exp(-i/200))
    T_Metropolis = 1/2 + np.sum(C_Metropolis)
    f = plt.figure(figsize=(9,6), dpi = 160)
    plt.plot(range(time), C_Metropolis, label = 'Metropolis')
    plt.plot(range(time), C_SwendsenWang, label = 'Swendsen Wang')
    plt.plot(range(time), C_Wolff, label = 'Wolff')
    plt.axhline(y=0.0, color = 'r', linestyle = '--', linewidth = 0.8)
    plt.ylim(-.2,1.1)
    plt.xlim(0,400)
    plt.legend(fontsize=14)
    plt.xlabel(r'$\tau$', fontsize = 14)
    plt.ylabel(r'$C(\tau)$', fontsize = 14)
    plt.title("Autocorrelation function of |M|", fontsize=16)
    plt.show()

M_Metropolis = [M_Metropolis[i]*15 for i in range(len(M_Metropolis))]
def plot_mag(M_Metropolis, M_SwendsenWang, M_Wolff):
    f = plt.figure(figsize=(8, 8), dpi=160, facecolor='w')
    sp =  f.add_subplot(2, 1, 1);
    plt.plot(range(10000), M_Metropolis[:10000], color='k', linewidth = 0.7)
    plt.title("Metropolis Update" , fontsize = 14)
    plt.xticks([])
    plt.xlim(0,10000)
    plt.ylabel('|M|', fontsize = 14)

    
#    plt.plot(range(10000), M_SwendsenWang[:10000])
#    plt.title("Swendsen-Wang Update")
#    plt.ylabel('|M|')
#    plt.show()
    sp =  f.add_subplot(2, 1, 2); 
    plt.plot(range(10000), M_Wolff[:10000], color='k', linewidth = 0.7)
    plt.title("Cluster Update", fontsize = 14)
    plt.ylabel('|M|' , fontsize = 14)
    plt.xlabel('state', fontsize = 14)
    plt.xlim(0,10000)
    plt
    plt.show()

plot_mag(M_Metropolis, M_SwendsenWang, M_Wolff)

    