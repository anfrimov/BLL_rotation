#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:14:52 2019

@author: anthony
"""
verbose = True
import cairo
#from datetime import datetime

#import seaborn as sn

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
def getneurons(df_cell, separators, convert=True):
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
    if not df.empty:
        grids = []
        if 'TimeStep' in df.columns:
            for area in range(12):
                grid = []
                for row in range(len(df)):
                    if df.TimeStep[row] == step and df.AreaAbs[row]-1 == area:
                        for column in columns:
                            neurons = getneurons(df[column][row],separators) 
                            grid.append(neurons)
                grids.append(grid)
            return grids
        else:
            for row in range(len(df)):
                grid = []
                for column in columns:
                    neurons = getneurons(df[column][row],separators)
                    grid.append(neurons)
                grids.append(grid)
            return grids
    else:
        return []

    
#%% Set pixel color
def setpixcol(ctx, grid_level, neuron, inputColor=[1.0,1.0,1.0]):
    if any([neuron == cell for cell in grid_level]):
        ctx.set_source_rgb(*inputColor)
                
    
#%% Draw grid on surface   
def drawgrid(ctx, gsize, grid, dist, buffer, size, mainColor, sleepColor=[0.2,0.2,0.2], showStatic=False, stat_grid=[[]], staticColor=[[]]):
    
    for y in range(gsize):
        for x in range(gsize):            
            ctx.set_source_rgb(*sleepColor)

            for level in range(len(stat_grid)):
                setpixcol(ctx, stat_grid[level], [x,y], staticColor[level])

            for level in range(len(grid)):
                setpixcol(ctx, grid[level], [x,y], mainColor[level])
                           
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

    
#%% Main script
if __name__ == '__main__':
    print('Importing main modules...')
    import re, os, getopt, sys
    import pandas as pd
    
    #%% Set defaults
    pprim = ''
    pstat = ''
    pdyna = ''
    
    imageFormat = 'png'
    outname = 'example'
    videoFormat = 'gif'
    fps = 5

    #%% Variables
    OutDir = os.getcwd() # sets output dir for script: default = cwd
    ImageDir = 'Images' # name of folder in which to store still frames and GIFs

    separators = re.compile('[#;]')
    data_columns = ['Neurons_Shared3', 
                    'Neurons_Shared2', 
                    'Neurons_Shared1', 
                    'Neurons_Shared0', 
                    'Shared', 
                    'Unique']

    showStatic = False

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

    ## Choose one static color:

    color_active = 'custom1'
    color_static = 'gray1-r'
    color_prim =   'main'


    # Define color choices
    # Same order as data_columns
    # Shared3, Shared2, Shared1, Shared0
    active_colors = {
            'custom1': [[0.141, 0.457, 0.8], [0.25, 0.629, 0.935], [0.424, 0.787, 1.0], [0.9, 0.5, 0.5]],
            'custom2': [[0.424, 0.787, 1.0], [0.25, 0.629, 0.935], [0.141, 0.457, 0.8], [0.9, 0.5, 0.5]],
            'blue1': [[0.886, 0.977, 1.0], [0.549, 0.861, 1.0], [0.317, 0.701, 0.972], [0.172, 0.517, 0.855]],
                #sn.cubehelix_palette(4, start=2.85, rot=-0.10, hue=2, dark=.45, light=.95),
            'blue2': [[0.772, 0.947, 1.0], [0.424, 0.787, 1.0], [0.206, 0.572, 0.897], [0.091, 0.336, 0.66]],
                #sn.cubehelix_palette(4, start=2.85, rot=-0.10, hue=2, dark=.3, light=.9),
            'iron1': [[0.968, 0.97, 0.82], [1.0, 0.619, 0.303], [1.0, 0.2, 0.4], [0.757, 0.006, 0.611]],
                #sn.cubehelix_palette(4, start=0.05, rot= 0.50, hue=3, dark=.30, light=.95),
            'iron2': [[1.0, 0.988, 0.875], [0.996, 0.765, 0.0], [0.945, 0.408, 0.012], [0.796, 0.075, 0.522]],
            'iron3': [[1.0, 0.929, 0.478], [1.0, 0.651, 0.224], [0.949, 0.353, 0.145], [0.812, 0.075, 0.502]],
            'blackbody1': [[0.991, 0.785, 0.36], [1.0, 0.453, 0.082], [1.0, 0.15, 0.098], [0.629, 0.0, 0.17]],
                #sn.cubehelix_palette(4, start=0.6, rot= 0.3, hue=3, dark=0.2, light=.8),
            'blackbody2': [[1.0, 0.988, 0.918], [1.0, 0.824, 0.0], [0.988, 0.345, 0.008], [0.769, 0.086, 0.145]]
            }
    # Shared3, Shared2, Shared1
    static_colors = {
            'gray1': [[0.702, 0.702, 0.702], [0.549, 0.549, 0.549], [0.4, 0.4, 0.4]],
                #sn.cubehelix_palette(3, hue=0, dark=.4, light=.7),
            'gray2': [[0.702, 0.702, 0.702], [0.451, 0.451, 0.451], [0.2, 0.2, 0.2]],
                #sn.cubehelix_palette(3, hue=0, dark=.2, light=.7),
            'gray1-r': [[0.4, 0.4, 0.4], [0.549, 0.549, 0.549], [0.702, 0.702, 0.702]],
                #sn.cubehelix_palette(3, hue=0, dark=.4, light=.7, reverse=True),
            'gray2-r': [[0.2, 0.2, 0.2], [0.451, 0.451, 0.451], [0.702, 0.702, 0.702]],
                #sn.cubehelix_palette(3, hue=0, dark=.2, light=.7, reverse=True),
            'blue1': [[0.424, 0.787, 1.0], [0.25, 0.629, 0.935], [0.141, 0.457, 0.8]],
                #sn.cubehelix_palette(3, start=2.85, rot=-0.10, hue=2, dark=.4, light=.7),
            'blue2': [[0.424, 0.787, 1.0], [0.172, 0.517, 0.855], [0.057, 0.219, 0.486]]
                #sn.cubehelix_palette(3, start=2.85, rot=-0.10, hue=2, dark=.2, light=.7)
            }
    # Shared, Unique
    prim_colors = {
            'main': [[0.1, 0.7, 0.1], [1, 1, 1]],
            'main-r': [[1, 1, 1], [0.1, 0.7, 0.1]]
            }
    
    disp_colors = False
    
    #%% Parse input arguments
    if verbose:
        print('Parsing input arguments...')
    try: 
        opts, args = getopt.gnu_getopt(sys.argv[1:],
                                   'ho:s:p:g:D:S:P:crn:',
                                   ['help', 
                                    'output=',
                                    'static=', 
                                    'primary=',
                                    'color-dynamic=',
                                    'color-static=',
                                    'color-primary=',
                                    'grid-size=',
                                    'display-colors',
                                    'palette-options',
                                    'reverse-bg',
                                    'mp4',
                                    'name=',
                                    'fps='])
    except getopt.GetoptError:
        print('Error: Unrecognized input or option. Use --help for details.')
        sys.exit(2)
    if args:
        pdyna = pd.read_csv(args[0], sep='\t')
    if len(args) > 1:
        print('Error: Too many arguments. Ending program.')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print('''
usage: 
    $ python neurogrids.py [<args>] [--help | -h] [--output | -o <output dir>] 
                           [--color-dynamic | -D <dynamic palette>]
                           [--static | -s <static pattern>] 
                           [--color-static | -S <static palette>] 
                           [--primary | -p <primary pattern>] 
                           [--color-primary | -P <primary palette>]
                           [--grid-size | -g <pixels>] [--display-colors] 
                           [--palette-options] [--reverse-bg] [--mp4]
                           [--name | -n <video filename>] [--fps <fps>]
  
arguments:
    Arguments can be included anywhere. The first argument must by the path to 
    dynamic_patterns.txt
    
options:
    --output, -o: 
        Set output directory for files generated by script (default: current 
        working directory).
    --static, -s:
        Provide path to static_patterns.txt, otherwise none will be generated.
    --primary, -p:
        Provide path to primary_patterns.txt, otherwise none will be generated.
    --palette-options:
        Show a list of available color palette presets for each type of 
        pattern. The names should be used in conjunction with -D, -S, -P.
    --color-dynamic, -D:
        Set color palette preset for dynamic patterns (see palette-options).
        Default is "custom1".
    --color-static, -S:
        Set color palette preset for static patterns (see palette-options).
        Default is "gray1-r"
    --color-primary, -P:
        Set color palette preset for dynamic patterns (see palette-options).
        Default is "main".
    --display-colors, -c:
        Show selected (or default) color palettes. Can be used in conjunction 
        with -D, -S, -P. If used with dynamic pattern input, will ask for
        confirmation of color choices before continuing.
    --grid-size, -g:
        Set the size of each grid in square pixels (default is 200).
    --reverse-bg:
        By default, the background color is black and inactive cells (neurons) 
        are dark gray. This reverses the colors so the background is dark gray 
        and inactive cells are black.
    --mp4:
        Outputs animation as an .mp4 file (default is .gif).
    --name, -n:
        Set filename of output animation (default is "example").
    --fps:
        Set the framerate of output animation file.

example call:
    
    $ python neurogrids.py input/dynamic_pattern.txt -D iron3 -S blue2 -r \\
      --mp4 --fps 10 -s input/static_pattern.txt -n example_call
    
                  ''')
            sys.exit()
        elif opt in ('-o','--output'):
            OutDir = arg
        elif opt in ('-s','--static'):
            pstat = pd.read_csv(arg, sep='\t')
            showStatic = True
        elif opt in ('-p','--primary'):
            pprim = pd.read_csv(arg, sep='\t')
        elif opt in ('-D','--color-dynamic'):
            color_active = arg
        elif opt in ('-S','--color-static'):
            color_static = arg
        elif opt in ('-P','--color-primary'):
            color_prim = arg
        elif opt in ('-g','--grid-size'):
            gsizePX = arg
        elif opt in ('-c','--display-colors'):
            disp_colors = True
        elif opt == '--palette-options':
            def all_colors(color_dict):
                color_list = ', '.join(list(color_dict.keys()))
                return(color_list)
            print('\n--color-dynamic (-D) options:\n',all_colors(active_colors))
            print('\n--color-static  (-S) options:\n',all_colors(static_colors))
            print('\n--color-primary (-P) options:\n',all_colors(prim_colors))
            sys.exit()
        elif opt in ('-r','--reverse-bg'):
            bgColor =       [0.2, 0.2, 0.2]
            sleepColor =    [0.0, 0.0, 0.0]
        elif opt == '--mp4':
            videoFormat = 'mp4'
        elif opt in ('-n','--name'):
            outname = arg
        elif opt == '--fps':
            fps = arg

    
    # Set colors to be used
    activeColor = active_colors[color_active]
    staticColor = static_colors[color_static]
    primColor = prim_colors[color_prim]

    ## For displaying selected palettes
    if disp_colors:
        print('Importing display modules...')
        import numpy as np
        import matplotlib.pyplot as plt
        colActive = activeColor[:]
        colStatic = [staticColor[0],staticColor[1],staticColor[2],sleepColor]
        colPrimar = [primColor[0],sleepColor,sleepColor,primColor[1]]
        
        all_colors = np.array([colActive,colStatic,colPrimar])
        
        fig, ax = plt.subplots()
        im = ax.imshow(all_colors)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(['Shared 3/Any', 'Shared 2', 'Shared 1', 'Unique'])
        ax.set_yticklabels(['Dynamic', 'Static', 'Primary'])
        ax.xaxis.tick_top()
        fig.tight_layout(pad=0)
        
        print('Figure open')
        plt.show(block=True)
        if args:
            answer = input('Continue with these colors? (y/n): ')
            if answer == 'n':
                print('User terminated program')
                sys.exit()
            elif not answer == 'n' and not answer == 'y':
                raise Exception('Input not recognized. Ending program.')
    if not args:
        print('No dynamic pattern specified. Ending program.\n(If this was unintentional, use -h or --help for details)')
        sys.exit()
        
    if verbose:
        print('Setting output directory...')
    ImageDir = os.path.join(OutDir,ImageDir)
    if not os.path.isdir(ImageDir):
        os.makedirs(ImageDir)
    if verbose:
        print('Output directory set to: %s' %(OutDir))
        
        
    #%% Check for static and primary patterns
    # Static
    if type(pstat) == pd.core.frame.DataFrame:
        stat_grids = makegrids(pstat, 'allts', data_columns[:3])
    else:
        stat_grids = []
        list4 = [[],[],[],[]]
        for i in range(12):
            stat_grids.append(list4)
    # Primary
    if type(pprim) == pd.core.frame.DataFrame:
        prim_grids = makegrids(pprim, None, data_columns[4:])
    else:
        prim_grids = []
    
    #%% Create images
    # Update output file name and generate .png image
    if verbose:
        print('Generating images...')
    
    for x in range(len(pdyna.TimeStep.unique())):
        # Set file name
        filenum = str(x+1)
        if len(filenum) < 2:
            filenum = '0' + filenum
        filename = os.path.join(ImageDir,'step_%s.%s' % (filenum,imageFormat))
        
        # Generate .png file
        grids = makegrids(pdyna, x+1, data_columns[:4])
        all_grids = [grids, stat_grids, prim_grids]
        makestep(*all_grids)
        if verbose:
            #sys.stdout.write('\r%i/%i completed.'%(x+1,len(pdyna.TimeStep.unique())))
            print('\r%i/%i completed.'%(x+1,len(pdyna.TimeStep.unique())),end='\r')

    #%% Create GIF from images
    if verbose:
        print('\nCreating animation from images...')
        
    import glob, imageio
    # Pull list of time step images from output directory    
    imageList = glob.glob(ImageDir+'/*.%s'%(imageFormat))
    filename = os.path.join(OutDir,'%s.%s'%(outname,videoFormat))
    
    fps = int(fps)
    settings = {
            'gif':{'duration':1/fps},
            'mp4':{'fps':fps, 'macro_block_size':None}
            }

    # Create GIF
    #with imageio.get_writer(filename, mode='I', duration=0.2) as writer:
    with imageio.get_writer(filename,mode='I',**settings[videoFormat]) as writer:
        for image in imageList:
            frame = imageio.imread(image)
            writer.append_data(frame)
    if videoFormat == 'mp4':
        print('This warning is only generated when creating an .mp4 file and can probably be ignored.')
    if verbose:
        print('Created "%s.%s" in "%s"'%(outname,videoFormat,OutDir))
        print('Ending program')