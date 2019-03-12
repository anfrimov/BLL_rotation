#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:14:52 2019

@author: anthony819
"""

import re, os, cairo, imageio, glob
import pandas as pd
import seaborn as sn

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

showStatic = True

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
#bgColor =       [0.2, 0.2, 0.2]
#sleepColor =    [0.0, 0.0, 0.0]

## Choose one static color:

color_active = 'custom1'
color_static = 'ungray1'
color_prim =   'main'


# Define color choices
active_colors = {
        'custom1': [[0.141, 0.457, 0.800],[0.250, 0.629, 0.935],[0.424, 0.787, 1.0],[.9,.5,.5]],
        'blue1': sn.cubehelix_palette(4, start=2.85, rot=-0.10, hue=2, dark=.45, light=.95),
        'blue2': sn.cubehelix_palette(4, start=2.85, rot=-0.10, hue=2, dark=.3, light=.9),
        'iron1': sn.cubehelix_palette(4, start=0.05, rot= 0.50, hue=3, dark=.30, light=.95),
        'iron2': [[1.000, 0.988, 0.875],[0.996, 0.765, 0.000],[0.945, 0.408, 0.012],[0.796, 0.075, 0.522]],
        'iron3': [[1.000, 0.929, 0.478],[1.000, 0.651, 0.224],[0.949, 0.353, 0.145],[0.812, 0.075, 0.502]],
        'blackbody1': sn.cubehelix_palette(4, start=0.6, rot= 0.3, hue=3, dark=0.2, light=.8),
        'blockbody2': [[1.0, 0.988, 0.918],[1.0, 0.824, 0.0],[0.988, 0.345, 0.008],[0.769, 0.086, 0.145]]
        }
static_colors = {
        'gray1': sn.cubehelix_palette(3, hue=0, dark=.4, light=.7),
        'gray2': sn.cubehelix_palette(3, hue=0, dark=.2, light=.7),
        'ungray1': sn.cubehelix_palette(3, hue=0, dark=.4, light=.7, reverse=True),
        'ungray2': sn.cubehelix_palette(3, hue=0, dark=.2, light=.7, reverse=True),
        'blue1': sn.cubehelix_palette(3, start=2.85, rot=-0.10, hue=2, dark=.4, light=.7),
        'blue2': sn.cubehelix_palette(3, start=2.85, rot=-0.10, hue=2, dark=.2, light=.7)
        }
prim_colors = {
        'main': [[0.1, 0.7, 0.1], [1, 1, 1]]
        }

# Set colors to be used
activeColor = active_colors[color_active]
staticColor = static_colors[color_static]
primColor = prim_colors[color_prim]

## For displaying selected palettes

#sn.palplot(activeColor)
#sn.palplot(staticColor)
#sn.palplot(primColor)

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
    
def getneurons(df_cell, convert=True):
    neurons = []
    if isinstance(df_cell, str):
        separator = separators.findall(df_cell)
        if separator:
            neurons = df_cell.split(sep=separator[0])
        else:
            neurons = [df_cell]
            
        while '' in neurons:
            neurons.remove('')
        if convert:
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
    if 'TimeStep' in df.columns:
        for area in range(12):
            grid = []
            for row in range(len(df)):
                if df.TimeStep[row] == step and df.AreaAbs[row]-1 == area:
                    for column in columns:
                        neurons = getneurons(df[column][row]) 
                        grid.append(neurons)
            grids.append(grid)
        return grids
    else:
        for row in range(len(df)):
            grid = []
            for column in columns:
                neurons = getneurons(df[column][row])
                grid.append(neurons)
            grids.append(grid)
        return grids


#%% Set pixel color

def setpixcol(ctx, grid_level, neuron, inputColor=[1.0,1.0,1.0]):
    
    if any([neuron == cell for cell in grid_level]):
        ctx.set_source_rgb(*inputColor)
        active = True
    else:
        active = False
        
    return active
                    

#%% Draw grid on surface
    
def drawgrid(ctx, gsize, grid, dist, buffer, size, mainColor, sleepColor=[0.2,0.2,0.2], showStatic=False, stat_grid=[[]], staticColor=[[]]):
    
    for y in range(gsize):
        for x in range(gsize):
            active = False; static = False;
            for level in range(len(grid)):
                active = setpixcol(ctx, grid[level], [x,y], mainColor[level])
                if active:
                    break
            
            if showStatic and not active:
                for level in range(len(grid)):
                    if level < len(stat_grid):
                        static = setpixcol(ctx, stat_grid[level], [x,y], staticColor[level])
                    if static:
                        break
                    
            if not active and not static:
                ctx.set_source_rgb(*sleepColor)
       
            ctx.rectangle(y*dist+buffer, x*dist+buffer, size, size)
            ctx.fill()


#%% Cairo image - create and save .png file for one time step
    
def makestep(grids, stat_grids, prim_grids):
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
    
    if not prim_grids:
        numCol = 6
        prim = 0
    else:
        numCol = 8
        prim = 1
        
    wd, ht = numCol*gsizePX+2*ibuff, 2*gsizePX+2*ibuff
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, wd, ht)
    ctx = cairo.Context(surface)
    
    ctx.scale(wd, ht) # Normalizing the canvas
    
    ctx.set_source_rgb(*bgColor)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    
    
    ctx.scale(1/numCol, 1/2)
    for z in range(len(grids)):
        ctx.set_matrix(cairo.Matrix(gsizePX, 
                                    0, 0, gsizePX, 
                                    (gsizePX*(z%6)+ibuff+gsizePX*prim), 
                                    (gsizePX*(z//6)+ibuff))) 
        
        drawgrid(ctx, gsize, grids[z], dist, buffer, size, activeColor, sleepColor, showStatic, stat_grids[z], staticColor)
    if numCol == 8:
        ctx.set_matrix(cairo.Matrix(gsizePX, 0, 0, gsizePX, ibuff, ibuff))
        drawgrid(ctx, gsize, prim_grids[0], dist, buffer, size, primColor, sleepColor)
        ctx.set_matrix(cairo.Matrix(gsizePX, 0, 0, gsizePX, (gsizePX*7+ibuff), ibuff))
        drawgrid(ctx, gsize, prim_grids[1], dist, buffer, size, primColor, sleepColor)
    
    surface.write_to_png(filename)


#%% Image testing
#showStatic = True
#
#size = 1/(gsize+sides)
#dist = 1/(gsize+sides) + 1/(gsize+sides)/gsize
#buffer = 1/(2*(gsize+sides))
#ibuff = int(round(gsizePX*buffer))
#    
#stat_grids = makegrids(pstat, 'allts', data_columns[:3])
#grids = makegrids(pdyna, 3, data_columns[:4])
#
#filename = 'test2.png'
#
#surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, gsizePX, gsizePX)
#ctx = cairo.Context(surface)
#
#ctx.scale(gsizePX, gsizePX) # Normalizing the canvas
#
#ctx.set_source_rgb(*bgColor)
#ctx.rectangle(0, 0, 1, 1)
#ctx.fill()
#
#drawgrid(ctx, gsize, grids[2], dist, buffer, size, activeColor, sleepColor, showStatic, stat_grids[2], staticColor)
#surface.write_to_png(filename)    

#%% Create images

# Update output file name and generate .png image
stat_grids = makegrids(pstat, 'allts', data_columns[:3])
prim_grids = makegrids(pprim, None, data_columns[4:])
    
for x in range(len(pdyna.TimeStep.unique())):
    filenum = str(x+1)
    if len(filenum) < 2:
        filenum = '0' + filenum
    filename = '/'.join([outputDir,'step_%s.png' % (filenum)])
    
    grids = makegrids(pdyna, x+1, data_columns[:4])
    makestep(grids, stat_grids, prim_grids)
    
#%% Create GIF from images

# Pull list of time step images from output directory    
imageList = glob.glob(outputDir+'/*.png')
filename = 'new.gif'

# Create GIF
with imageio.get_writer(filename, mode='I', duration=0.2) as writer:
    for image in imageList:
        frame = imageio.imread(image)
        writer.append_data(frame)
         