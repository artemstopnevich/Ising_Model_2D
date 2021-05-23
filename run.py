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
Metropolis =  True
# Modes:
MetropolisObservables = True
MetropolisBinder = False

# Swendsen-Wang Cluster Algo
SW = False
# Modes:
SWObservables = True
SWSim = False
SWBinder = False

# Wolff Cluster Algo
W = False
# Modes:
WObservables = True
WSim = False
WBinder = False

if Metropolis == True:
    print("2D Ising Model: Metropolis Algorithm")
    if MetropolisObservables == True:
        print("mode: Physical Observables")
        import Metropolis.run_metropolis
    if MetropolisBinder == True:
        print("mode: Binder Ratios")
        import Metropolis.run_binder_metropolis
        
if SW == True:
    print("2D Ising Model: Swendsen-Wang Cluster Algorithm")
    if SWObservables == True:
        print("mode: Physical Observables")
        import SwendsenWang.swendsenwang_cluster
    if SWSim == True:
        print("mode: Epochs simulation")
        import SwendsenWang.swendsenwang_cluster_sim
    if SWBinder == True:
        print("mode: Binder Ratios")
        import SwendsenWang.swendsenwang_binder

if W == True:
    print("2D Ising Model: Wolff Cluster Algorithm")
    if WObservables == True:
        print("mode: Physical Observables")
        import Wolff.wolff_cluster
    if WSim == True:
        print("mode: Epochs simulation")
        import Wolff.wolff_cluster_sim
    if WBinder == True:
        print("mode: Binder Ratios")
        import Wolff.wolff_binder
        