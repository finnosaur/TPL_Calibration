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

def set_galvos(volts_xy):
    
    core.set_property(tpl_X, 'Voltage', volts_xy[0])
    core.set_property(tpl_Y, 'Voltage', volts_xy[1])
    print('Setting X,Y voltages to: {vx}, {vy} volts'.format(vx = volts_xy[0], vy= volts_xy[1]))
    
    return

def get_galvo_voltages():
    
    v_x = core.get_property('NIDAQAO-Dev3/ao0-Galvo-X-Axis', 'Voltage')
    v_y = core.get_property('NIDAQAO-Dev3/ao1-Galvo-Y-Axis', 'Voltage')
    
    volt_vals = [v_x, v_y]
    
    return np.array(volt_vals)

def pix_to_volts_empirical(xy_pos, func ,x_popt, y_popt):
    
    x = xy_pos[0]
    y = xy_pos[1]
    
    vx = func( (x, y) , *x_popt)  
    vy =func( (x, y) , *y_popt) 
    
    return [vx, vy]



# =============================================================================
# # please input the directory for your current project
# =============================================================================
project_dir = r'C:\Local_Data\Finn\20240122_pycromanager'


# =============================================================================
# End of user input
# =============================================================================

save_dir = os.path.join(project_dir, 'TPL_Calibration')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print('Made directory %s' %save_dir) 


core = Core()

camera = core.get_camera_device()

print("Current CORE CAM = " + str(camera))

core.set_camera_device('Camera-2')
camera = core.get_camera_device()
print("Setting CORE CAM = " + str(camera))
core.set_exposure(camera, 100) # setting exposur to 100 ms


# definte property names for galvo and TPL control
tpl_X = 'NIDAQAO-Dev3/ao0-Galvo-X-Axis'
tpl_Y = 'NIDAQAO-Dev3/ao1-Galvo-Y-Axis'
tpl_shutter = 'NIDAQAO-Dev3/ao3-TPL-ChBlank'



#%% Snap a test image
core.set_property(tpl_shutter, 'Voltage', 5)


cur_im = snap_image()
plt.imshow(cur_im, cmap='gray')


core.set_property(tpl_shutter, 'Voltage', 0)

print('do you see a good spot? If not, do not continue with callibration.')

#%% rough calibration
# =============================================================================
# For x galvo
# =============================================================================

# these are rough estimages for lims of FOV 
# you must find them empirically by adjusting the galvo voltage manually
v_x_min = -2.25
v_x_max = 2
v_step = 0.5

v_y_min = -2
v_y_max = 2.5



# make range of voltage values
x_voltages = np.arange(v_x_min, v_x_max, v_step)
y_voltages = np.arange(v_y_min, v_y_max, v_step)


volt_xpos_arr = np.zeros((len(x_voltages), 3)) # init array for v_x, x, y

core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 5)
# loop thru x and y separately
# set y to middle value
v_y_0 = (v_y_max + v_y_min)*0.5
i = 0
for v_x in x_voltages:
    
    # move the mirrors
    print('Setting X,Y voltages to: {vx}, {vy} volts'.format(vx = v_x, vy= v_y_0))
    
    core.set_property(tpl_X, 'Voltage', v_x)
    core.set_property(tpl_Y, 'Voltage', v_y_0)
    # time.sleep(0.1)
    
    # snap an image 
    cur_im = snap_image()
    save_name = os.path.join(save_dir, 'vx_cal_{}.tif'.format(i))
    
    # save the image
    # imageio.imwrite(save_name, cur_im)
    
    # find the spot
    f = tp.locate(cur_im, 5, minmass=50)
    
    loc = f.to_numpy()
    
    x_coord = loc[0,1] # get the x coord in pix
    y_coord = loc[0,0] # get y coord in pix
    spot_pos = np.array([x_coord,y_coord])
    
    tp.annotate(f, cur_im)
    
    
    volt_xpos_arr[i,:] = np.append(v_x, spot_pos)
    
    i +=1
    
    
    

# close the 2PL shutter    
core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 0)

# =============================================================================
# # repeat fourgh calibration for y 
# =============================================================================

#open TPL shutter
core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 5)
volt_ypos_arr = np.zeros((len(y_voltages), 3)) # init array for v_x, x, y

# loop thru x and y separately
# set y to middle value
v_x_0 = (v_x_max + v_x_min)*0.5
i = 0
for v_y in y_voltages:
    
    # move the mirrors
    print('Setting X,Y voltages to: {vx}, {vy} volts'.format(vx = v_x_0, vy= v_y))
    
    core.set_property(tpl_X, 'Voltage', v_x_0)
    core.set_property(tpl_Y, 'Voltage', v_y)
    # time.sleep(0.1)
    
    # snap an image 
    cur_im = snap_image()
    save_name = os.path.join(save_dir, 'vy_cal_{}.tif'.format(i))
    
    # # save the image
    # # imageio.imwrite(save_name, cur_im)
    
    # # find the spot
    f = tp.locate(cur_im, 5, minmass=50)
    
    loc = f.to_numpy()
    
    x_coord = loc[0,1] # get the x coord in pix
    y_coord = loc[0,0] # get y coord in pix
    spot_pos = np.array([x_coord,y_coord])
    
    tp.annotate(f, cur_im)
    
    
    volt_ypos_arr[i,:] = np.append(v_y, spot_pos)
    
    i +=1
    
    
    

# close the 2PL shutter    
core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 0)


#%% Linea regression fit for rough calibration

# =============================================================================
# # x voltages
# =============================================================================
x = volt_xpos_arr[:,0]

y = volt_xpos_arr[:,1]

x_coefs = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(x_coefs) 
# poly1d_fn is now a function which takes in x and returns an estimate for y


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('X Spot Pos vs X Galvo Voltage')
ax1.set (xlabel = 'X Galvo Volts (V)', ylabel= 'X Position (pix)')
ax1.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker




# =============================================================================
# Y voltages
# =============================================================================
x = volt_ypos_arr[:,0]

y = volt_ypos_arr[:,2]

y_coefs = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(y_coefs) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

ax2.set_title('Y Spot Pos vs Y Galvo Voltage')
ax2.set( xlabel =  'Y Galvo Volts (V)', ylabel = 'Y Position (pix)')
ax2.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

# save plot
plt.savefig(os.path.join(save_dir,'rough_cal_lin_fit.jpg'))

print('Rought calibration fits saved to{}'.format(save_dir))

#%% Advanced calibration

# lets make a square grid of points
# lims of square grid 
grid_min = 300
grid_max = 800
n = 5 # nxn points in a grid


xlin = np.linspace(grid_min, grid_max, n)
ylin = np.linspace(grid_min, grid_max, n)

my_grid = np.meshgrid(xlin, ylin)

x_test= my_grid[0].flatten()
y_test = my_grid[1].flatten()

# sweep through grid using our linear models
# these models give us a corase ability to loclize
# we will use this grid to create a more accurate empirical function over the domain
# of the grid
core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 5)
volt_pos_arr2 = np.zeros((len(x_test), 4)) # array will store detected xpos, detected ypos, x voltage, y voltage
for i in range(len(x_test)):
    
    spot_pos = [x_test[i],y_test[i]]
    
    volt_vals = pix_to_volts(spot_pos, x_coefs, y_coefs)
    
    set_galvos(volt_vals)
    
    cur_im = snap_image()
    
    f = tp.locate(cur_im, 5, minmass=50)

    loc = f.to_numpy()

    x_coord = loc[0,1] # get the x coord in pix
    y_coord = loc[0,0] # get y coord in pix
    spot_pos = np.array([x_coord,y_coord])
    tp.annotate(f, cur_im)
    
    cur_volt_vals = get_galvo_voltages()
    
    volt_pos_arr2[i,:] = np.append(spot_pos, cur_volt_vals)
    
    print(volt_pos_arr2[i,:])
    


core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 0)
print('2PL Shutter Closed') 

#%% plot voltages vs spot detection coords as scatter, perform fit

xpos = volt_pos_arr2[:,0]
ypos = volt_pos_arr2[:,1]
xvolt = volt_pos_arr2[:,2]
yvolt = volt_pos_arr2[:,3]


# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
# Creating plot
ax.scatter(xpos, ypos, xvolt)
 
# show plot
plt.show()

# fit the surfaces

# Perform curve fitting 


# fit function is imported from a separate file to ensure continuity 

# get vx fit
vx_popt, vx_pcov = curve_fit(func, (xpos, ypos), xvolt) 

# get vy fit
vy_popt, vy_pcov = curve_fit(func, (xpos, ypos), yvolt) 


# Print and save optimized parametersfor the xvolt vs x,y
print('~~~~~')
print('Fit params for x votlage saved')
print(vx_popt)
np.savetxt(os.path.join(save_dir, 'x_popt.csv'), vx_popt, delimiter= ',')
print(os.path.join(save_dir, 'x_popt.csv'))
os.path.join(save_dir, 'y_popt.csv')
print('~~~~~')
# Print optimized parametersfor the xvolt vs x,y
print('~~~~~')
print('Fit params for y votlage saved')
print(vy_popt)
np.savetxt(os.path.join(save_dir, 'y_popt.csv'), vy_popt, delimiter= ',')
print(os.path.join(save_dir, 'y_popt.csv'))
print('~~~~~')

# lets plot over a mesh (same one we created for the scan)
X, Y = np.meshgrid(xlin, ylin) 

# evaluate over mesh for both
xvolt_pred = func( (X, Y) , *vx_popt)

yvolt_pred = func( (X, Y) , *vy_popt) 


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
# =============
# First subplot
# =============

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(xpos, ypos, xvolt)

ax.set_xlabel('X (pix)') 
ax.set_ylabel('Y (pix)') 
ax.set_zlabel('X voltage (V)')
ax.set_title('X Galvo Voltage vs (x,y)')


surf_xvolt = ax.plot_surface(X, Y, xvolt_pred, rstride=1, cstride=1, color='red', alpha=0.5,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf_yvolt = ax.plot_surface(X, Y, yvolt_pred, rstride=1, cstride=1, color='yellow', alpha=0.5,
                        linewidth=0, antialiased=False)
ax.scatter(xpos, ypos, yvolt)

ax.set_xlabel('X (pix)') 
ax.set_ylabel('Y (pix)') 
ax.set_zlabel('Y voltage (V)') 

ax.set_title('Y Galvo Voltage vs (x,y)')



plt.show()


#%%  we neet to now evaluate targetting accuracy using these models

# define a grid within the lims of the calibration area
# lets make a square grid of points
# lims of square grid 
grid_min = 400
grid_max = 800
n = 10  # nxn points in a grid

xlin = np.linspace(grid_min, grid_max, n)
ylin = np.linspace(grid_min, grid_max, n)

my_grid = np.meshgrid(xlin, ylin)
x_test= my_grid[0].flatten()
y_test = my_grid[1].flatten()


# fig, ax = plt.subplots()

plt.scatter(x_test, y_test, alpha = 0.2, c = 'gray')


core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 5)
loc_resids_arr = np.zeros((len(x_test), 6)) # array will store desired x, desired y, detected x, detected y
for i in range(len(x_test)):
    
    test_pos = np.array([x_test[i],y_test[i]])
    
    volt_vals = pix_to_volts_empirical(test_pos, func, vx_popt, vy_popt)
    
    set_galvos(volt_vals)
    
    cur_im = snap_image()
    
    f = tp.locate(cur_im, 5, minmass=50)

    loc = f.to_numpy()

    x_coord = loc[0,1] # get the x coord in pix
    y_coord = loc[0,0] # get y coord in pix
    spot_pos = np.array([x_coord,y_coord])
    
    # ax.scatter(x_coord, y_coord, facecolors='none')
    # ax.imshow(cur_im)
    
    # plt.sca(ax)
    tp.annotate(f, cur_im, plot_style={'markersize': 4, 'linewidth': 0.3})
    
    resid = np.subtract(test_pos, spot_pos)
    
    loc_resids_arr[i,:] = np.concatenate((test_pos, spot_pos, resid), axis = None)
    
    print('Progress = ' + str(i/len(x_test)))
    
    
core.set_property('NIDAQAO-Dev3/ao3-TPL-ChBlank', 'Voltage', 0)   
set_galvos(pix_to_volts_empirical([600,600], func, vx_popt, vy_popt))

print('~~~~~~~~~~~~~~~~')
print('fine calibration complete')
print('TPL set to pixel [600,600]')
print('TPL shutter closed')
print('~~~~~~~~~~~~~~~~')


#%%
# plot residuals as root mean square error

dr = np.sqrt(np.square(loc_resids_arr[:,4]) + np.square(loc_resids_arr[:,5]) )

rmse = np.sqrt(np.square(dr).mean())
    
pix_size = 109

dr_grid = pix_size* np.reshape(dr, np.shape(my_grid[0]))


x_labels = np.ceil(xlin)
y_labels = np.ceil(ylin)

#plotting the heatmap 
hm = sn.heatmap(data = dr_grid, xticklabels=x_labels, yticklabels=y_labels, annot= False, 
                cbar_kws={'label': 'dr (nm)'}) 
hm.set_title('Residuals heatmap')
hm.set_xlabel('X target coord (pix)')

hm.set_ylabel('Y target coord (pix)')

plt.savefig(os.path.join(save_dir,'fine_calibration_heatmap.jpg'));  
# displaying the plotted heatmap 
plt.show()



print('~~~~~~~~~~~~~~~~')
print('Heat map saved to')
print(os.path.join(save_dir,'fine_calibration_heatmap.jpg'));
print('~~~~~~~~~~~~~~~~')

