#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:14:52 2019

@author: anthony819
"""

import re, os, cairo, imageio, glob
import pandas as pd

pprim = pd.read_csv('Inputs/01_primary_areas_patterns.txt', sep='\t')
pstat = pd.read_csv('Inputs/02_static_patterns.txt', sep='\t')
pdyna = pd.read_csv('Inputs/03_dynamic_patterns.txt', sep='\t')

#%% Variables

wdir = os.listdir(os.getcwd()) # sets working dir for script: default = cwd
outputDir = 'Images' # name of folder in which to store still frames and GIFs

separators = re.compile('[#;]')
data_columns = ['Neurons_Shared3', 
                'Neurons_Shared2', 
                'Neurons_Shared1', 
                'Neurons_Shared0', 
                'Shared', 
                'Unique']

#%% Cairo Image Variables
            
# sizes
# desired square space (in pixels) occupied by a single grid / brain region
# counting grid buffer
gsizePX = 200

gsize = 25 # grid size in number of cells, square
sides = 2 # spacing buffer for cells


# colors
bgColor =       [0.0, 0.0, 0.0] # black
sleepColor =    [0.2, 0.2, 0.2] # dark gray

# green+purple
#activeColor =   [[0.9, 0.2, 1.0],
#                 [0.7, 0.4, 0.9],
#                 [0.5, 0.6, 0.8],
#                 [0.3, 0.8, 0.7]]

# CYM+red
activeColor =   [[0.0, 1.0, 1.0],
                 [1.0, 1.0, 0.0],
                 [1.0, 0.0, 1.0],
                 [0.5, 0.0, 0.0]]

# iron, mine
##activeColor =   [[1.000, 0.929, 0.478],
#                 [1.000, 0.651, 0.224],
#                 [0.949, 0.353, 0.145],
#                 [0.812, 0.075, 0.502]]

# iron, calc
#activeColor = [[1.000, 0.988, 0.875],
#               [0.996, 0.765, 0.000],
#               [0.945, 0.408, 0.012],
#               [0.796, 0.075, 0.522]] 

# glowbow
#activeColor = [[1.0, 0.988, 0.918],
#               [1.0, 0.824, 0.0],
#               [0.988, 0.345, 0.008],
#               [0.769, 0.086, 0.145]] 
            

#%% Array to grid converter
    
def array2grid(num, size=25):
    """Returns row and column coordinates of an element in an array if it
    were instead located in a square matrix of the given size.
    
    num (int)= index number of array element
    size (int)= length of one dimension of square matrix
    
    [row, col] (int)= row and column numbers of element
    """
    num = int(num)
    num -= 1
    row = num//size
    col = num%size
    return [row,col]

#%% Get neurons
    
def getneurons(df_cell):
    neurons = []
    if isinstance(df_cell, str):
        separator = separators.findall(df_cell)
        if separator:
            neurons = df_cell.split(sep=separator[0])
        else:
            neurons = [df_cell]
            
        while '' in neurons:
            neurons.remove('')
        for n in range(len(neurons)):
            neurons[n] = array2grid(neurons[n])
        
    return neurons
#%% Generate neural activity grid and transform
    
def makegrids(df, step, columns):
    """Creates list of active neuron grids for a given time step: necessary
    data for a single image / animation frame.
    
    df (DataFrame)= pandas DataFrame loaded from .csv file
    step (int)= desired time step, starting with 1
    
    grids (list)= list containing 12 lists of active neurons, one for each
        region
    """
    
    grids = []
    for area in range(12):
        grid = []
        for row in range(len(df)):
            if df.TimeStep[row] == step and df.AreaAbs[row]-1 == area:
                for column in columns:
                    neurons = getneurons(df[column][row]) 
                    grid.append(neurons)
        grids.append(grid)
    return grids

#%% Cairo image - create and save .png file for one time step
    
def makesteps(grids):
    """Generates .png image of a given time step.
    
    grids (list)= Output from makegrids()
    """

    # size:     relative cell length
    # dist:     space taken up by cell and its cell buffer
    # buffer:   grid buffer size
    # ibuff:    buffer size in pixels
    size = 1/(gsize+sides)
    dist = 1/(gsize+sides) + 1/(gsize+sides)/gsize
    buffer = 1/(2*(gsize+sides))
    ibuff = int(round(gsizePX*buffer))
    
    wd, ht = 6*gsizePX+2*ibuff, 2*gsizePX+2*ibuff
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, wd, ht)
    ctx = cairo.Context(surface)
    
    ctx.scale(wd, ht) # Normalizing the canvas
    
    ctx.set_source_rgb(*bgColor)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    
    
    ctx.scale(1/6, 1/2)
    for z in range(len(grids)):
        ctx.set_matrix(cairo.Matrix(gsizePX, 
                                    0, 0, gsizePX, 
                                    (gsizePX*(z%6)+ibuff), 
                                    (gsizePX*(z//6)+ibuff))) 
    
        for y in range(gsize):
            for x in range(gsize):
                for level in range(len(grids[z])):
                    if any([[x,y] == cell for cell in grids[z][level]]):
                        ctx.set_source_rgb(*activeColor[level])
                        break
                    else:
                        ctx.set_source_rgb(*sleepColor)
                    
                ctx.rectangle(y*dist+buffer, x*dist+buffer, size, size)
                ctx.fill()
    
    surface.write_to_png(filename)


#%% Create images

# Update output file name and generate .png image
for x in range(len(pdyna.TimeStep.unique())):
    filenum = str(x+1)
    if len(filenum) < 2:
        filenum = '0' + filenum
    filename = '/'.join([outputDir,'step_%s.png' % (filenum)])
    
    grids = makegrids(pdyna, x+1, data_columns[:4])
    makesteps(grids)
    
#%% Create GIF from images

# Pull list of time step images from output directory    
imageList = glob.glob(outputDir+'/*.png')
filename = 'new.gif'

# Create GIF
with imageio.get_writer(filename, mode='I', duration=0.2) as writer:
    for image in imageList:
        frame = imageio.imread(image)
        writer.append_data(frame)
         