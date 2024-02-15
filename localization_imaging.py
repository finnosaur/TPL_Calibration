from mpl_toolkits import mplot3d
import seaborn as sn
import numpy as np
import time
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import os
import trackpy as tp
import skimage as sk
from pycromanager import Core
from pycromanager import Acquisition, multi_d_acquisition_events
import matplotlib.pyplot as plt
import imageio
from scipy.optimize import curve_fit 
from matplotlib import cm
from tpl_functions import *

from func import func

# where images will be saved
save_dir = r'C:\Local_Data\Finn\20240122_pycromanager'

# pycromanager set up 
core = Core()

camera = core.get_camera_device()
print("CORE CAM = " + str(camera))

# TPL 
tpl_shutter_status = core.get_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage')
tpl_shutter = 'NIDAQAO-Dev3/ao3-TPL-ChBlank'
tpl_state = 0

# definte property names for galvo control
tpl_X = 'NIDAQAO-Dev3/ao0-Galvo-X-Axis'
tpl_Y = 'NIDAQAO-Dev3/ao1-Galvo-Y-Axis'


# we need to load the optimized fit params genereated from callibration
x_popt =  np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'x_popt.csv'))
y_popt =  np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'y_popt.csv'))
print('Loaded fit params from: ' +os.path.join(os.path.dirname(os.path.realpath(__file__)) ))


# =============================================================================
# Run
# =============================================================================

#%% detect a locus

# set channel with spot
core.set_camera_device('Camera-2')
core.set_exposure(camera, 100) # setting exposur to 100 ms
#snap pic
cur_im = snap_image()
plt.imshow(cur_im)

# track
f = tp.locate(cur_im, 5, minmass=50)
loc = f.to_numpy()

x_coord = loc[0,1] # get the x coord in pix
y_coord = loc[0,0] # get y coord in pix
spot_pos = np.array([x_coord,y_coord])

# visualize
tp.annotate(f, cur_im)



#%% localize TPL to spot
set_tpl_shutter(tpl_shutter, 1)

volt_vals = pix_to_volts_empirical(spot_pos, func, x_popt, y_popt)

set_galvos(volt_vals)

# cur_im = snap_image()



set_tpl_shutter(tpl_shutter, 0)

#%% This script can only be run after running TPL calibration script
set_tpl_shutter(tpl_shutter, 1)
test_pos = [600,600]

volt_vals = pix_to_volts_empirical(test_pos, func, x_popt, y_popt)

set_galvos(volt_vals, tpl_X, tpl_Y)

cur_im = snap_image()

f = tp.locate(cur_im, 5, minmass=50)

loc = f.to_numpy()

x_coord = loc[0,1] # get the x coord in pix
y_coord = loc[0,0] # get y coord in pix
spot_pos = np.array([x_coord,y_coord])

# plt.sca(ax)
tp.annotate(f, cur_im, plot_style={'markersize': 15, 'linewidth': 0.3})

print(spot_pos)
set_tpl_shutter(tpl_shutter, 0)

#%%

tpl_state = tpl_switch(tpl_shutter, tpl_state)

#%%


#%%

with Acquisition(directory=save_dir, name='test') as acq:
    events = multi_d_acquisition_events(num_time_points=10)
    acq.acquire(events)