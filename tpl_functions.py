import seaborn as sn
import numpy as np
import time
import numpy as np
import os
import trackpy as tp
from pycromanager import Core
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from func import func
from datetime import datetime
import tpl_functions as tpl
# =============================================================================
# Define functions! 
# =============================================================================

core = Core()


# swithces the TPL AOM shutter on or off
def tpl_switch(tpl_shutter, tpl_state):
    
    if tpl_state == 1:
        
        core.set_property(tpl_shutter, 'Voltage', 0)
        tpl_state = 0;
        print('TPL Shutter Closed')
        
    else:
        core.set_property(tpl_shutter, 'Voltage',5)
        tpl_state = 1;
        print('TPL Shutter Open')
        
    return tpl_state

def set_tpl_shutter(tpl_shutter, state):
    
    if state == 1:
        core.set_property(tpl_shutter, 'Voltage', 5)
        print('TPL Shutter Open')
    elif state == 0:
        core.set_property(tpl_shutter, 'Voltage', 0)
        print('TPL Shutter Closed')
        
        
# sends voltage values to uManager (tpl_x and )
def set_galvos(volts_xy, tpl_x, tpl_y):
    
    core.set_property(tpl_x, 'Voltage', volts_xy[0])
    core.set_property(tpl_y, 'Voltage', volts_xy[1])
    print('Setting X,Y voltages to: {vx}, {vy} volts'.format(vx = volts_xy[0], vy= volts_xy[1]))
    
    return

# use our fit function to convert from desired pixel coords to voltages
def pix_to_volts_empirical(xy_pos, func ,x_popt, y_popt):
    
    x = xy_pos[0]
    y = xy_pos[1]
    
    vx = func( (x, y) , *x_popt)  
    vy =func( (x, y) , *y_popt) 
    
    return [vx, vy]

def get_galvo_voltages(tpl_x, tpl_y):
    
    v_x = core.get_property(tpl_x, 'Voltage')
    v_y = core.get_property(tpl_y, 'Voltage')
    
    volt_vals = [v_x, v_y]
    
    return np.array(volt_vals)

# snap a single image usnig current camera
def snap_image():
    # acquire an image and display it
    core.snap_image()
    tagged_image = core.get_tagged_image()
    # get the pixels in numpy array and reshape it according to its height and width
    image_array = np.reshape(
        tagged_image.pix,
        newshape=[-1, tagged_image.tags["Height"], tagged_image.tags["Width"]],
    )
    # for display, we can scale the image into the range of 0~255
    image_array = (image_array / image_array.max() * 255).astype("uint8")
    # return the first channel if multiple exists
    return image_array[0, :, :]

def pix_to_volts(spot_pos, x_coefs, y_coefs):
    
    xpos = spot_pos[0]
    ypos = spot_pos[1]
    
    x_volt = (xpos - x_coefs[1]) / x_coefs[0]
    y_volt = (ypos - y_coefs[1]) / y_coefs[0]
    
    volts_xy = np.array([x_volt, y_volt])
    return(volts_xy)

if __name__ == "__main__":
    print('hi')