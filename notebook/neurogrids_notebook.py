#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:14:52 2019

@author: anthony
"""

import re, os, sys, glob, cairo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Image


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
            df_step = df[df.TimeStep == step]
            for area in range(1,13):
                grid = []
                df_area = df_step[df_step.AreaAbs == area]
                for column in columns:
                    neurons = getneurons(df_area[column].loc[df_area.index[0]],separators)
                    grid.append(neurons)
                grids.append(grid)
            return grids
        else:
            for row in range(len(df)):
                grid = []
                for column in columns:
                    neurons = getneurons(df[column].loc[row],separators)
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
                if level <= len(staticColor) - 1:
                    setpixcol(ctx, stat_grid[level], [x,y], staticColor[level])

            for level in range(len(grid)):
                setpixcol(ctx, grid[level], [x,y], mainColor[level])
                           
            ctx.rectangle(y*dist+buffer, x*dist+buffer, size, size)
            ctx.fill()


#%% Cairo image - create and save .png file for one time step   
def makestep(grids, stat_grids, prim_grids, filename):
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
ImageDir = os.path.join(OutDir,'Images') # name of folder in which to store still frames and GIFs

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
        'custom1': [[0.424, 0.787, 1.0], [0.25, 0.629, 0.935], [0.141, 0.457, 0.8], [0.9, 0.5, 0.5]],
        'custom1-r': [[0.141, 0.457, 0.8], [0.25, 0.629, 0.935], [0.424, 0.787, 1.0], [0.9, 0.5, 0.5]],
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



# Layout
widget_lay = widgets.Layout(width='60%')
subui_lay = widgets.Layout(width='90%')
html_lay = widgets.Layout(position='center', width='35%')
test_lay = widgets.Layout(width='32%')
check_lay = widgets.Layout(width='90%', position='left', margin='0px 0px 0px 0px')
check_html_lay = widgets.Layout(width='85%', position='right', margin='0px 0px 0px 0px')

## Select colors widget
# active
active_widget = widgets.Dropdown(options=active_colors, layout=widget_lay)
active_title = widgets.HTML('Dynamic:', layout=html_lay)
active_ui = widgets.HBox([active_title, active_widget])#, layout=subui_lay)

# static
static_widget = widgets.Dropdown(options=static_colors, layout=widget_lay)
static_title = widgets.HTML('Static:', layout=html_lay)
static_ui = widgets.HBox([static_title, static_widget])#, layout=subui_lay)

# primary
primar_widget = widgets.Dropdown(options=prim_colors, layout=widget_lay)
primar_title = widgets.HTML('Primary:', layout=html_lay)
primar_ui = widgets.HBox([primar_title, primar_widget])#, layout=subui_lay)

# reverse background colors
reverseBG_widget = widgets.ToggleButton(description='Reverse BG')
reverse_title = widgets.HTML('%s'%(reverseBG_widget.value), 
                             layout=widgets.Layout(padding='0px 0px 0px 10px'))
def update_reverse(*args):
    reverse_title.value = '%s'%(reverseBG_widget.value)
reverseBG_widget.observe(update_reverse, 'value')

reverse_ui = widgets.HBox([reverseBG_widget, reverse_title])

# display palette function
def show_palette(Active, Static, Primary, Background):
    # Set colors to be used
    global activeColor, staticColor, primColor, sleepColor, bgColor
    activeColor = active_widget.value
    staticColor = static_widget.value
    primColor   = primar_widget.value
    
    if reverseBG_widget.value:
        sleepColor =    [0.0, 0.0, 0.0] # black
        bgColor    =    [0.2, 0.2, 0.2] # dark gray
    else:
        bgColor    =    [0.0, 0.0, 0.0] # black
        sleepColor =    [0.2, 0.2, 0.2] # dark gray

    ## For displaying selected palettes
    colActive = activeColor[:]
    colStatic = [staticColor[0],staticColor[1],staticColor[2],sleepColor]
    colPrimar = [primColor[0],sleepColor,sleepColor,primColor[1]]

    all_colors = np.array([colActive,colStatic,colPrimar])
    
    # Print selected values
    #print(active_widget.label, static_widget.label, primar_widget.label, reverseBG_widget.value)
    
    # Show palette
    fig, ax = plt.subplots()
    im = ax.imshow(all_colors)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['Shared 3/Any', 'Shared 2', 'Shared 1', 'Unique'])
    ax.set_yticklabels(['Dynamic', 'Static', 'Primary'])
    ax.xaxis.tick_top()
    fig.tight_layout(pad=0)
    
#palette = interactive(show_palette, Active=active_widget, Static=static_widget, 
#                      Primary=primar_widget, Background=reverseBG_widget)

palette = widgets.interactive_output(show_palette, {'Active':active_widget, 
                                     'Static':static_widget, 'Primary':primar_widget, 
                                     'Background':reverseBG_widget})
palette.layout.width = '540px'

#display(ui, palette)
# hide palette
def show_palette(pal_show):
    if pal_show:
        palette.layout.visibility = 'visible'
    else:
        palette.layout.visibility = 'hidden'
hide_pal_wid = widgets.interactive(show_palette, 
                                   pal_show=widgets.Checkbox(True, description='Show Palette', 
                                                             indent=False, layout=check_lay))

# full ui
color_ui = widgets.VBox([active_ui, static_ui, primar_ui, reverse_ui, hide_pal_wid], layout=test_lay)

full_color_ui = widgets.HBox([color_ui, palette])#, layout=subui_lay)
#display(full_color_ui)

main_prim = pd.read_csv('Inputs/01c_primary_areas_patterns.txt', sep='\t')
main_stat = pd.read_csv('Inputs/02c_static_patterns.txt', sep='\t')
main_dyna = pd.read_csv('Inputs/03c_dynamic_patterns.txt', sep='\t')

models = main_dyna.Model.unique()
types = main_dyna.Type.unique()
patterns = main_dyna.Pattern.unique()
#models = list(set(main_stat.Model))
#types = list(set(main_stat.Type))
#patterns = list(set(main_prim.Pattern))

#for i in [models, types, patterns]:
#    i.sort()

#model_wid = widgets.Dropdown(options=models, description='Model:')
model_wid = widgets.Dropdown(options=models, layout=widget_lay)
model_title = widgets.HTML('Model:', layout=html_lay)
model_ui = widgets.HBox([model_title, model_wid])#, layout=subui_lay)

#type_wid = widgets.Dropdown(options=types, description='Type:')
type_wid = widgets.Dropdown(options=types, layout=widget_lay)
type_title = widgets.HTML('Type:', layout=html_lay)
type_ui = widgets.HBox([type_title, type_wid])

#pattern_wid = widgets.Dropdown(options=patterns, description='Pattern:')
pattern_wid = widgets.Dropdown(options=patterns, layout=widget_lay)
pattern_title = widgets.HTML('Pattern:', layout=html_lay)
pattern_ui = widgets.HBox([pattern_title, pattern_wid])

def reduce_df(df, model, model_type, pattern):
    if 'Model' in df.columns:
        df = df[df.Model == model]
    df = df[df.Type == model_type]
    df = df[df.Pattern == pattern]
    return df

def reduce_dfs(model, model_type, pattern):
    global pprim, pstat, pdyna
    #pprim = reduce_df(main_prim, model_wid.value, type_wid.value, pattern_wid.value)
    #pstat = reduce_df(main_stat, model_wid.value, type_wid.value, pattern_wid.value)
    #pdyna = reduce_df(main_dyna, model_wid.value, type_wid.value, pattern_wid.value)
    pprim = reduce_df(main_prim, model, model_type, pattern)
    pstat = reduce_df(main_stat, model, model_type, pattern)
    pdyna = reduce_df(main_dyna, model, model_type, pattern)
    #display(pprim.head(), pstat.head(), pdyna.head())
    #return pprim, pstat, pdyna

reduce = widgets.interactive_output(reduce_dfs, {'model':model_wid, 'model_type':type_wid, 'pattern':pattern_wid})
#reduce = widgets.interact_manual(reduce_dfs, model=model_wid, model_type=type_wid, pattern=pattern_wid)
#display(reduce)

#input_ui = widgets.VBox([model_wid, type_wid, pattern_wid])
#input_ui = widgets.VBox([model_ui, type_ui, pattern_ui])

#display(input_ui,reduce)

if not os.path.isdir(ImageDir):
    os.makedirs(ImageDir)

verbose=False

show_stat_wid = widgets.Checkbox(indent=False, description='Show Static', layout=check_lay)
#show_stat_title = widgets.HTML('Show Static', layout=check_html_lay)
#show_stat_ui = widgets.HBox([show_stat_title, show_stat_wid])

show_prim_wid = widgets.Checkbox(indent=False, description='Show Primary', layout=check_lay)
#show_prim_title = widgets.HTML('Show Primary', layout=check_html_lay)
#show_prim_ui = widgets.HBox([show_prim_title, show_prim_wid])

progress = widgets.FloatProgress(
    value=0,
    min=0,
    max=1,
    step=0.001,
    description='Load Bar',
    bar_style='info',
    orientation='horizontal'
)

progress.layout.visibility = 'hidden'

def create_pngs(b):
    global pprim, pstat, pdyna
    pprim = reduce_df(main_prim, model_wid.value, type_wid.value, pattern_wid.value)
    pstat = reduce_df(main_stat, model_wid.value, type_wid.value, pattern_wid.value)
    pdyna = reduce_df(main_dyna, model_wid.value, type_wid.value, pattern_wid.value)
    
    if os.path.isdir(ImageDir):
        files = glob.glob(os.path.join(ImageDir,'*'))
        if files:
            for file in files:
                os.remove(file)
                
    #%% Check for static and primary patterns
    # Static
    if show_stat_wid.value:
        stat_grids = makegrids(pstat, 'allts', data_columns[:3])
    else:
        stat_grids = []
        list4 = [[],[],[],[]]
        for i in range(12):
            stat_grids.append(list4)
    # Primary
    if show_prim_wid.value:
        prim_grids = makegrids(pprim, None, data_columns[4:])
    else:
        prim_grids = []
    
    #%% Create images
    # Update output file name and generate .png image
    if verbose:
        print('Generating images...')
    steps = len(pdyna.TimeStep.unique())
    progress.description = 'Loading'
    progress.layout.visibility = 'visible'
    for x in range(steps):
        # Set file name
        filename = os.path.join(ImageDir,'step_%.2i.%s' % (x+1,imageFormat))
        progress.value = (x+1)/steps

        # Generate .png file
        grids = makegrids(pdyna, x+1, data_columns[:4])
        all_grids = [grids, stat_grids, prim_grids, filename]
        makestep(*all_grids)
        if verbose:
            #sys.stdout.write('\r%i/%i completed.'%(x+1,len(pdyna.TimeStep.unique())))
            print('\r%i/%i completed.'%(x+1,len(pdyna.TimeStep.unique())),end='\r')
    progress.description = 'Complete'
    progress.layout.visibility = 'hidden'



#input_ui = widgets.VBox([model_wid, type_wid, pattern_wid, show_stat_wid, show_prim_wid], layout=test_lay)
input_ui = widgets.VBox([model_ui, type_ui, pattern_ui, show_stat_wid, show_prim_wid], layout=test_lay)

full_ui = widgets.HBox([input_ui, color_ui, palette])
gen_images = widgets.Button(description='Generate Images')
gen_images.on_click(create_pngs)

def test_fun(a, b):
    print(show_stat_wid.value)
    print(show_prim_wid.value)

test = widgets.interactive_output(test_fun, {'a':show_stat_wid, 'b':show_prim_wid})

#display(full_ui, gen_images, progress)

#final_choice = widgets.interact_manual(create_pngs, 
#                                       model=model_wid, 
#                                       model_type=type_wid, 
#                                       pattern=pattern_wid, 
#                                       show_static=show_stat_wid, 
#                                       show_primary=show_prim_wid)

def show_pngs(play):
    if os.path.isdir(ImageDir):
        images = glob.glob(os.path.join(ImageDir,'*'))
        if not images:
            img = Image('step_00.png')
        else:
            img = Image(images[play-1])
        display(img)
        
play = widgets.Play(
    interval=200,
    value=1,
    min=1,
    max=30,
    step=1,
    description="Press play",
    disabled=False
)
play._repeat = True

play_slider = widgets.IntSlider(min=1, max=30)

fps = widgets.FloatText(
    value=5,
    description='FPS:',
    disabled=False
)

def click_fstep(b):
    play._playing = False
    if play.value == 30:
        play.value = 1
    else:
        play.value += 1

def click_bstep(b):
    play._playing = False
    if play.value == 1:
        play.value = 30
    else:
        play.value -= 1
    
fstep = widgets.Button(
    description='Step ->',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me'
)
bstep = widgets.Button(
    description='<- Step',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me'
)

fstep.on_click(click_fstep)
bstep.on_click(click_bstep)

def update_interval(*args):
    play.interval = 1000 / fps.value
fps.observe(update_interval, 'value')

widgets.jslink((play, 'value'), (play_slider, 'value'))
full_play = widgets.HBox([play, play_slider, bstep, fstep])    


output = widgets.interactive_output(show_pngs, {'play':play})

#display(output, full_play, fps)

gen_line = widgets.HBox([gen_images, progress])

neurogrids = [full_ui, gen_line, output, full_play, fps]