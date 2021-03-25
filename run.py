#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:14:50 2021

@author: artemstopnevich
"""

'''
    Master file
    -----------
    Choose, which option to run below
'''

# Metropolis Algo - enable with (Metropolis = True) and choose the mode
Metropolis = False
# Modes:
MetropolisObservables = False
MetropolisBinder = False

# Swendsen-Wang Cluster Algo
Cluster = True
# Modes:
ClusterObservables = False
ClusterSim = False
ClusterBinder = True


if Cluster == True:
    print("2D Ising Model: Cluster Algorithm")
    if ClusterObservables == True:
        print("mode: Physical Observables")
        import SwendsenWang.run_cluster
    if ClusterSim == True:
        print("mode: Epochs simulation")
        import SwendsenWang.run_cluster_sim
    if ClusterBinder == True:
        print("mode: Binder Ratios")
        import SwendsenWang.run_binder

if Metropolis == True:
    print("2D Ising Model: Metropolis Algorithm")
    if MetropolisObservables == True:
        print("mode: Physical Observables")
        import Metropolis.run_metropolis
    if MetropolisBinder == True:
        print("mode: Binder Ratios")
        import Metropolis.run_binder_metropolis