#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:38:50 2021

@author: artemstopnevich
"""


# Simulating the Ising model

from __future__ import division
import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt


class Ising():
    
    def __init__(self, Ns, nt, eqSteps, mcSteps, temp=None, mcTime=None):
        self.Ns       = Ns
        self.nt       = nt
        self.eqSteps  = eqSteps
        self.mcSteps  = mcSteps    
        self.temp     = temp 
        self.mcTime   = mcTime
        
    
    def InitialState(self, N):   
        '''generates a random spin configuration for initial condition'''
        config = 2*np.random.randint(2, size=(N,N))-1
        return config

    def equilibrate(self, S, beta, N, eqSteps):
        for i in range(eqSteps):
            for ii in range(N*N):
                a = np.random.randint(0,N)
                b = np.random.randint(0,N)
                nb = S[(a+1)%N,b] + S[a,(b+1)%N] + S[(a-1)%N,b] + S[a,(b-1)%N] 
                dE = 2*nb*S[a,b]
                if dE < 0:	
                    S[a,b] *= -1
                elif rand() < np.exp(-dE*beta):
                    S[a,b] *= -1
        return 0
    
    def mcstep(self, S, N, beta):
        ''' This is to execute the monte carlo iterations using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        for ii in range(N*N):
            a = np.random.randint(0,N)
            b = np.random.randint(0,N)
            nb = S[(a+1)%N,b] + S[a,(b+1)%N] + S[(a-1)%N,b] + S[a,(b-1)%N] 
            dE = 2*nb*S[a,b]
            if dE < 0:	
                S[a,b] *= -1
            elif rand() < np.exp(-dE*beta):
                S[a,b] *= -1
        return 0
    
    def calcEnergy(self, S, N):
        '''Energy of a given configuration'''
        energy = 0
        for i in range(1,N):
            for j in range(1,N):
                nb = S[(i-1)%N, j] + S[i,(j-1)%N]
                energy += -nb*S[i,j]
        return energy
    
    
    def calcMag(self, S, N):
        '''Magnetization of a given configuration'''
        mag = 0
        for i in range(1,N):
            for j in range(1,N):
                mag += S[i,j]
        return mag   
    
    def Block(self, step, t0, A, mcSteps, blockA):
        sigma = 0
        if step > 0 and step%t0==0:
            blockA.append(A/step)
        if step+1 == mcSteps:
            meanA = np.mean(blockA)
            sigma_block = np.sqrt((1/len(blockA)) * sum([(blockA[z] - meanA)**2 for z in range(len(blockA))]))
            sigma = sigma_block/np.sqrt(len(blockA))
        return sigma
        
    def autocorrelation(self, step, A, mcSteps, stepA):
        gamma=[]
        dt=100
        while step < mcSteps:
            stepA.append(A/step)
        if step+1 == mcSteps:
            meanA = np.mean(stepA)
        for j in range(len(stepA)):
            top = 0
            bottom = 0
            for i in range(len(stepA)-j):
                top += (stepA[i] - meanA)*(stepA[i+j] - meanA)
                bottom += (stepA[i] - meanA)**2
            gamma[j] = top/bottom
        t0 = dt/2 + np.sum(gamma*dt)
        return t0
    
    def configData(self, S, E, M, C, X, B, C_T, M_T, Error_E, Error_M, Error_C):
        eqSteps = self.eqSteps
        mcSteps = self.mcSteps
        N = self.Ns
        nt = self.nt
        # divide by number of samples, and by system size to get intensive values
        n1 = 1.0/(mcSteps*N*N)
        n2 = 1.0/(mcSteps*mcSteps*N*N) 

        for t in range(nt):                           
            E1 = M1 = E2 = M2 = 0
            blockE=[]
            blockE2=[]
            blockM=[]
            blockM2=[]
            stepE=[]
            iT=B[t] 
            iT2=iT*iT
            
            S = self.InitialState(N)
            self.equilibrate(S, iT, N, eqSteps)
        
            for i in range(mcSteps):
                self.mcstep(S, N, iT)           
                Ene = self.calcEnergy(S, N)    
                Mag = self.calcMag(S, N)      
        
                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag 
                E2 = E2 + Ene*Ene
                
                #errors
#                t0 = self.autocorrelation(i, E1, mcSteps, stepE)
                t0 = 100
                sigma_E= self.Block(i, t0, E1, mcSteps, blockE)
                sigma_E2= self.Block(i, t0, E2, mcSteps, blockE2)
#                t0 = self.autocorrelation(i, M1, mcSteps, stepA)
                sigma_M= self.Block(i, t0, M1, mcSteps, blockM)
                sigma_M2= self.Block(i, t0, M2, mcSteps, blockM2)                
                
            
            E[t] = n1*E1
            M[t] = n1*M1
#            MD[t] = n2*M1
            C[t] = (n1*E2 - n2*E1*E1)*iT2
            X[t] = (n1*M2 - n2*M1*M1)*iT
            Error_E[t] = sigma_E
            Error_M[t] = sigma_M
            Error_C[t] = (sigma_E2 - sigma_E*sigma_E)*iT2
            
            
            T_c = 2/math.log(1 + math.sqrt(2))
            coeff = math.log(1 + math.sqrt(2))
            if 1.0/iT - T_c >= 0:
                C_T[t] = 0
            else:
                M_T[t] = pow(1- pow(np.sinh(2*iT), -4), 1/8)
                C_T[t] = (2.0/np.pi) * (coeff**2) * (-math.log(1-1.0/(iT*T_c)) + math.log(1.0/coeff) - (1 + np.pi/4)) 
            # creating dictionaries
        return E, M, C, X, C_T, M_T, Error_E, Error_M, Error_C
            
    def simulate(self, N, temp, mcTime):   
        ''' This module simulates the Ising model at a given temperature'''
        config = self.InitialState(self.N)
        f = plt.figure(figsize=(10, 10), dpi=80)  
        f.suptitle("Snapshots of the configurations of {}x{} lattice at T = {}".format(N,N,temp), fontsize = 14)
        self.configPlot(f, config, 0, N, 1)
        beta = 1.0/temp
        for i in range(mcTime):
            self.mcstep(config, beta)
            if i == 1:       self.configPlot(f, config, i, N, 2)
            if i == 4:       self.configPlot(f, config, i, N, 3)
            if i == 32:      self.configPlot(f, config, i, N, 4)
            if i == 128:     self.configPlot(f, config, i, N, 5)
            if i == 1000:    self.configPlot(f, config, i, N, 6)
            if i > 1001: 
                if i ==mcTime-1: self.configPlot(f, config, i, N, 7)
        f.savefig("Results/snapshot_N-{}_t-{}.png".format(self.N,temp))
        plt.show()        
                 
                    
    def configPlot(self, f, config, i, N, n):
        ''' This modules plots the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(N), range(N))
        sp =  f.add_subplot(3, 3, n)  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap = plt.cm.Accent);
        plt.title('Time=%d'%i); plt.axis('tight')    
        
    def CriticalTemp(self, dims, nt, eqSteps, mcSteps):
        minTemp = 2.2
        maxTemp = 2.3
        Magnetizations = self.configData(dims, minTemp, maxTemp, nt, eqSteps, mcSteps)[-1]
        lattices = np.array(list(Magnetizations.keys()))
        data = dict()
        for i in range(1,len(lattices)):
            logs = (np.log(abs(Magnetizations[lattices[0]]))/ np.log(abs(Magnetizations[lattices[i]])))/ np.log(dims[0]/dims[i])
        
            data[lattices[i]] = logs
        return lattices, data
    
    def plotTemp_C(self, lattices, nt, data):
        Temperature = np.linspace(2.2, 2.3, nt)
        for i in range(2, len(lattices)):
            f = data[lattices[i]]
            g = data[lattices[i-1]]
            plt.plot(Temperature, data['{}'.format(lattices[i-1])])
            idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
            print(Temperature[idx])
        plt.show()
            
           
        

    
