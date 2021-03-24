#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:30:26 2021

@author: artemstopnevich
"""


import numpy as np


def blocking_error(A, num_blocks):
    '''
        perform error calculation by spliting up
        data in a series of blocks
        :param A: variable (array of all mcStep points)
        :param num_blocks: number of blocks to split into
        :return: total error for the variable @temp point
    '''
    mean_values = np.zeros(num_blocks)
    error_block = np.zeros(num_blocks)
    itemize = range(1, num_blocks+1)
    blocks = np.array_split(A, num_blocks)
    global_mean = np.mean(A)
    for i in range(num_blocks):
        mean_values[i] = np.mean(blocks[i])
        error_block = np.sqrt((1/itemize[i]) * np.sum((mean_values - global_mean)**2))
    error = error_block/np.sqrt(num_blocks)

    return error

