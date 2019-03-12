#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:07:53 2019

@author: anthony819
"""
print('\rImporting re...',end='\r')
import re
print('\rImporting re, pandas...',end='\r')
import pandas as pd
print('\rImporting re, pandas, neurogrids...')
from neurogrids import array2grid, getneurons
print('Import complete')

#%% Collect neurons
    
def collect_neurons(df, dynamic=True):
    if dynamic:
        ranges = range(4)
    else:
        ranges = range(1,4)
        
    basic_dict = {}
    for region in range(1,13):
        basic_dict[region] = {}
        for shared in ranges:
            level = 'Neurons_Shared%i'%(shared)
            basic_dict[region][level] = []
            for row in range(len(df)):
                if df.AreaAbs[row] == region:
                    basic_dict[region][level].extend(getneurons(df[level][row],separators,convert=False))
            basic_dict[region][level] = list(set(basic_dict[region][level]))

    return basic_dict


#%% Test for differences
print('Compiling separators (# and ;)')
separators = re.compile('[#;]')
dyna = '../../Inputs/03b_dynamic_patterns.txt'
stat = '../../Inputs/02b_static_patterns.txt'
print('Reading patterns from:\n\t%s\n\t%s'%(dyna,stat))
pdyna = pd.read_csv(dyna,sep='\t')
pstat = pd.read_csv(stat,sep='\t')
print('Calculating...')
pdyna_all = collect_neurons(pdyna)
pstat_all = collect_neurons(pstat,False)

for region in pdyna_all:
    for shared in range(1,4):
        level = 'Neurons_Shared%i'%(shared)
        for neuron in pdyna_all[region][level]:
            truth = neuron in pstat_all[region][level]
            if not truth:
                print('Region: %i'%(region))
                print('Level: %s'%(level))
                print('\t%s = %s'%(neuron,str(array2grid(neuron))))
                for location in pstat_all[region]:
                    other = neuron in pstat_all[region][location]
                    if other:
                        print('\tWrong shared level: %s\n'%(location))
                        break
                if not other:
                    print('\tNo static found\n')
print('Done')