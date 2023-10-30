#--------------------------------------------------------------------
#--------------------------------------------------------------------

#           Code to obtain the turnover frequency from 5 maps
#
#           The code needs the 5 maps aligned on their respective
#           origins (core component or brightest pixel). The core 
#           shift is performed with parameters xshift and yshift

#                 The code has also a version with four maps

#           Check "user input" for the parameters that have to be 
#           modified by the user

#           Check "important" for some important info on what the code does

#           Outputs of the code: 
#           i)   Plot of the brightness together with the fitted spectrum for a single pixel (to use as a control);
#           ii)  Turnover frequency map (as contours the highest frequency map);
#           iii) Higher turnover frequency map (highest possible turnover frequency inferred from the fit with the error);
#           iv)  Lower turnover frequency map (lowest possible turnover frequency inferred from the fit with the error);
#           v)   Chi square map;
#           vi)  Map of the brightness at the turnover frequency;

#--------------------------------------------------------------------
#--------------------------------------------------------------------

#Import libraries

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from numpy import log as ln
from scipy.optimize import curve_fit

#Import usefull commands

from astropy.io import fits
from math import sqrt,pi
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import Cutout2D
from numpy import log as ln
from scipy.optimize import curve_fit

#Define the plot canvas

fig, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = [8,6]


#Turnover frequency function (synchrotron spectrum, 
# optically thick index fixed to 2.5 according to Lobanov, 1999)

def func_turn(x,I0,turn,alpha0):
    alphat = 2.5
    return I0 * (x/turn)**alphat * (1.0 - np.exp( - (turn/x)**(alphat - alpha0)) )

# ------------------------------------------
# Start main user input
# ------------------------------------------

FITS_file_1 = 'NGC315_5_BP_middleturn.fits'
nu_1 = 5.          #frequency (in GHz)
rms_1 = 0.038e-3   #rms noise (in Jy/beam)
err_1 = 0.05       #error on the brightness (in %)
xshift1 = 4        #how many pixel of shit from the core at the highest frequency (x coord)? (File_5)
yshift1 = 4        #how many pixel of shit from the core at the highest frequency (y coord)? (File_5)

FITS_file_2 = 'NGC315_8_BP_middleturn.fits'
nu_2 = 8.4         #frequency (in GHz)
rms_2 = 0.063e-3   #rms noise (in Jy/beam)
err_2 = 0.05       #error on the brightness (in %)
xshift2 = 3        #how many pixel of shit from the core at the highest frequency (x coord)? (File_5)
yshift2 = 3        #how many pixel of shit from the core at the highest frequency (y coord)? (File_5)

FITS_file_3 = 'NGC315_15_NOBP_June23_middleturn.fits'
nu_3 = 15.3        #frequency (in GHz)
rms_3 = 0.067e-3   #rms noise (in Jy/beam)
err_3 = 0.05       #error on the brightness (in %)
xshift3 = 1        #how many pixel of shit from the core at the highest frequency (x coord)? (File_5)
yshift3 = 1        #how many pixel of shit from the core at the highest frequency (y coord)? (File_5)

FITS_file_4 = 'NGC315_22_NOBP_middleturn.fits'
nu_4 = 22.2        #frequency (in GHz)
rms_4 = 0.096e-3   #rms noise (in Jy/beam)
err_4 = 0.05       #error on the brightness (in %)
xshift4 = 1        #how many pixel of shit from the core at the highest frequency (y coord)? (File_5)
yshift4 = 1        #how many pixel of shit from the core at the highest frequency (y coord)? (File_5)

FITS_file_5 = 'NGC315_43_NOBP_middleturn.fits'
nu_5 = 43.1        #frequency (in GHz)
rms_5 = 0.542e-3   #rms noise (in Jy/beam)
err_5 = 0.10       #error on the brightness  (in %)
maxi = 0.298       #brightest pixel in Jy/beam - for plotting

fr = [nu_1, nu_2, nu_3, nu_4, nu_5]   #array of frequencies

thres = 10.0            #threshold for rms cut (the pixels below thres*brightness will be set to 0)
px_size = 0.16          #mas per pixel
conv_pc = 0.331         #conversion from mas to pc (to change accordingly to the source)
x = 1024                #x size of the maps (pixel)
y = 1024                #y size of the maps (pixel)
position = (515, 515)   #position of the center (x,y)
size = (40,40)          #size of the output images

# ------------------------------------------
# End main user input
# ------------------------------------------

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#                       Computing the turnover frequency map
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# ---------- Open the .fits file number 1 ----------

Intensity_map1 = fits.open(FITS_file_1)
Coord1 = get_pkg_data_filename(FITS_file_1)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ---------- Open the .fits file number 2 ----------

Intensity_map2 = fits.open(FITS_file_2)
Coord2 = get_pkg_data_filename(FITS_file_2)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map2.info()

#Save the data in a numpy array
image_data2 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data2[i][j] = Intensity_map2[0].data[0][0][i][j]

# ---------- Open the .fits file number 3 ----------

Intensity_map3 = fits.open(FITS_file_3)
Coord3 = get_pkg_data_filename(FITS_file_3)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map3.info()

#Save the data in a numpy array
image_data3 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data3[i][j] = Intensity_map3[0].data[0][0][i][j]

# ---------- Open the .fits file number 4 ----------

Intensity_map4 = fits.open(FITS_file_4)
Coord4 = get_pkg_data_filename(FITS_file_4)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map4.info()

#Save the data in a numpy array
image_data4 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data4[i][j] = Intensity_map4[0].data[0][0][i][j]
        
# ---------- Open the .fits file number 5 ----------

Intensity_map5 = fits.open(FITS_file_5)
Coord5 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map5.info()

#Save the data in a numpy array
image_data5 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data5[i][j] = Intensity_map5[0].data[0][0][i][j]

# ---------- Compute the turnover map ----------

#Initialize 2D arrays for maps

turnover         = [[0 for _ in range(x)] for _ in range(y)]
turnover_valp    = [[0 for _ in range(x)] for _ in range(y)]
turnover_valm    = [[0 for _ in range(x)] for _ in range(y)]
turn_over_fl     = [[0 for _ in range(x)] for _ in range(y)]
err_turn         = [[0 for _ in range(x)] for _ in range(y)]
turn_over_fl_err = [[0 for _ in range(x)] for _ in range(y)]
chi_square       = [[0 for _ in range(x)] for _ in range(y)]

#Initialize arrays for storing info

distance   = []
turn_rid   = []
turn_p_rid = []
turn_m_rid = []
flux_rid   = []
flux_p_rid = []
flux_m_rid = []

print('---------------------------------------------------------')

for j in range(10,y-10):
    for i in range(10,x-10):
        
        bright     = []
        err_bright = []
        array_val  = []
        
        #fill in arrays
        
        if(image_data1[i-xshift1][j-yshift1] > thres*rms_1):
            bright.append(image_data1[i-xshift1][j-yshift1])
            err_bright.append(image_data1[i-xshift1][j-yshift1]*err_1)
            
        if(image_data2[i-xshift2][j-yshift2] > thres*rms_2):
            bright.append(image_data2[i-xshift2][j-yshift2])
            err_bright.append(image_data2[i-xshift2][j-yshift2]*err_2)
            
        if(image_data3[i-xshift3][j-yshift3] > thres*rms_3):        
            bright.append(image_data3[i-xshift3][j-yshift3])
            err_bright.append(image_data3[i-xshift3][j-yshift3]*err_3)
            
        if(image_data4[i-xshift4][j-yshift4] > thres*rms_4):        
            bright.append(image_data4[i-xshift4][j-yshift4])
            err_bright.append(image_data4[i-xshift4][j-yshift4]*err_4)
            
        if(image_data5[i][j] > thres*rms_5):
            bright.append(image_data4[i][j])
            err_bright.append(image_data5[i][j]*err_5)
        
        #IMPORTANT: the code analyzes only the pixels in which all the maps have points 
        
        if(len(bright) == 5):
            popt, pcov = curve_fit(func_turn, fr, bright, sigma = err_bright, maxfev = 1000000)
            perr = np.sqrt(np.diag(pcov))
            chi_sq = 0.0
            for p in range(0,len(bright)):
                chi_sq = chi_sq + (bright[p] - func_turn(fr[p],*popt))**2.0 / (err_bright[p]**2.0)
            chi_square[i][j] = chi_sq
            
            # parameters for finding values

            high_nu = 43.1    # user input - highest frequency
            low_nu = 4.9      # user input - lower frequency
            x_p       = np.linspace(low_nu,high_nu,1000)
            
    
            array_val              = func_turn(x_p,*popt)
            peak                   = max(array_val)
            turnover_index         = (np.where( array_val == peak ))[0]
            turnover_val           = (low_nu + (high_nu - low_nu)/1000.*turnover_index)[0]
            turnover[i][j]         = turnover_val
            turn_over_fl[i][j]     = peak
                
            
            if( math.isnan(perr[0]) == False and math.isnan(perr[1]) == False and math.isnan(perr[2]) == False 
               and perr[0]/perr[0] == 1.0 and perr[1]/perr[1] == 1.0 and perr[2]/perr[2] == 1.0):
                array_val              = func_turn(x_p,popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
                if(math.isnan(max(array_val)) == False):
                    peakp                  = max(array_val)
                    turnover_index         = (np.where( array_val == peakp ))[0]
                    turnover_val_mom       = (low_nu + (high_nu - low_nu)/1000.*turnover_index)[0]
                    turnover_valp[i][j]    = abs(turnover_val - turnover_val_mom)
                    turnover_valpa         = turnover_val_mom - turnover_val
                else:
                    turnover_valp[i][j]    = 0.0
             
                array_val              = func_turn(x_p,popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2])
                if(math.isnan(max(array_val)) == False):
                    peakm                  = max(array_val)
                    turnover_index         = (np.where( array_val == peakm ))[0]
                    turnover_val_mom       = (low_nu + (high_nu - low_nu)/1000.*turnover_index)[0]
                    turnover_valm[i][j]    = abs(turnover_val - turnover_val_mom)
                    turnover_valma         = turnover_val_mom - turnover_val
                else:
                    turnover_valm[i][j]    = 0.0

                    
            # filling out arrays
    
            array_val              = func_turn(x_p,*popt)
            peak                   = max(array_val)
            turnover_index         = (np.where( array_val == peak ))[0]
            turnover_val           = (low_nu + (high_nu - low_nu)/1000.*turnover_index)[0]
                
                
            #use this to check some specific pixels
                    
            if(i == 515 and j == 514):  #user input: here insert the coordinates of the pixel you want to see the fitting
                print('Intensity real = ' + str(peak))
                print('Turnover real = ' + str(turnover_val))
                X = [turnover_val, turnover_val]
                Y = [0.0, peak]
                plt.errorbar(X,Y,color='black',linestyle='dashed', alpha=0.7)
                X = [0.0, turnover_val]
                Y = [peak, peak]
                plt.errorbar(X,Y,color='black',linestyle='dashed', alpha=0.7)
                ax.errorbar(fr, bright, yerr=err_bright, fmt='o', color='blue')
                plt.plot(x_p,func_turn(x_p,*popt), color='green')
                plt.plot(x_p,func_turn(x_p,popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2]), color='green', linestyle='dashed', alpha=0.7)
                plt.plot(x_p,func_turn(x_p,popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2]), color='green', linestyle='dashed', alpha=0.7)
                axes = plt.gca()
                axes.xaxis.label.set_size(16)
                axes.yaxis.label.set_size(16)
                ax.tick_params(axis='x', labelsize=16)
                ax.tick_params(axis='y', labelsize=16)
                plt.xlim(4.5, 43.5) #user input: modify accordingly
                plt.ylim(0.25, 0.5) #user input: modify accordingly
                plt.xlabel('Frequency [GHz]')
                plt.ylabel('Brightness [Jy/beam]')
                #plt.legend(loc='lower right', ncol=1, handleheight = 1.0, labelspacing=0.005, prop={'size':12})
                filename = 'single_profile_turnover_514_515.png' #user input: modify accordingly
                plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
            
            #if turnover outside of the frequency range set to zero
            
            if(turnover_val > (nu_5 - 1) or turnover_val < (nu_1 + 1) ):
                turnover[i][j]     = 0.0
                err_turn[i][j]     = 0.0 
                chi_square[i][j]   = 0.0
                turn_over_fl[i][j] = 0.0
        if(len(bright) != 5):
            turnover[i][j]     = 0.0
            err_turn[i][j]     = 0.0
            chi_square[i][j]   = 0.0
            turn_over_fl[i][j] = 0.0
         
# ---------- Plotting the turnover frequency map ----------


# ---------- Save the .fits for the turnover map ----------

hdu.data = turnover
SM_filename = 'Turnover_map.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- Set the info for the plotting ----------

fig = plt.figure()
mpl.rcParams.update({'font.size': 35})
plt.tick_params(axis='both', labelsize = 45)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Relative right ascension [mas]', fontsize=30)
plt.ylabel('Relative declination [mas]', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover frequency [GHz]')
fig.set_figheight(13)
fig.set_figwidth(13)

# ---------- Open the contour map (map5) ----------

Intensity_map1 = fits.open(FITS_file_5)
Coord1 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ----------- Create sub-images --------------------

cutout = Cutout2D(image_data1, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
plt.contour(hdu.data_c, levels=[-0.012*maxi, 0.012*maxi,
                                0.024*maxi, 0.048*maxi, 0.096*maxi, 0.192*maxi,
                                0.384*maxi, 0.768*maxi, 0.95*maxi], colors='white')
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------

plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                   Plot the higher limit map
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# ---------- Save the .fits for the turnover frequency map ----------

hdu.data = turnover_valp
SM_filename = 'Turnover_error_map_plus.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- Set the info for the plotting ----------

fig = plt.figure()
mpl.rcParams.update({'font.size': 35})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet, vmax = 20.)
plt.xlabel('Right ascension [mas]', fontsize=30)
plt.ylabel('Declination [mas]', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover frequency error [GHz]')
fig.set_figheight(15)
fig.set_figwidth(15)

# ---------- Open the contour map (map5) ----------

Intensity_map1 = fits.open(FITS_file_5)
Coord1 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ----------- Create sub-images --------------------

cutout = Cutout2D(image_data1, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
plt.contour(hdu.data_c, levels=[-0.012*maxi, 0.012*maxi,
                                0.024*maxi, 0.048*maxi, 0.096*maxi, 0.192*maxi,
                                0.384*maxi, 0.768*maxi, 0.95*maxi], colors='black',
                                extent=[0,40.*0.16,0,40.*0.16])
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------

plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map_error.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                   Plot the lower limit map
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# ---------- Save the .fits for the turnover frequency map ----------

hdu.data = turnover_valm
SM_filename = 'Turnover_error_map_minus.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- Set the info for the plotting ----------

fig = plt.figure()
mpl.rcParams.update({'font.size': 35})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Right ascension [pixel]', fontsize=30)
plt.ylabel('Declination [pixel]', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover frequency error [GHz]')
fig.set_figheight(15)
fig.set_figwidth(15)

# ---------- Open the contour map (map5) ----------

Intensity_map1 = fits.open(FITS_file_5)
Coord1 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ----------- Create sub-images --------------------

cutout = Cutout2D(image_data1, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
plt.contour(hdu.data_c, levels=[-0.012*maxi, 0.012*maxi,
                                0.024*maxi, 0.048*maxi, 0.096*maxi, 0.192*maxi,
                                0.384*maxi, 0.768*maxi, 0.95*maxi], colors='black')
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------

plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map_error.pdf',format='pdf', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                          Chi square map
# -----------------------------------------------------------------
# -----------------------------------------------------------------


# ---------- Save the .fits for the turnover frequency map ----------

hdu.data = chi_square
SM_filename = 'Turnover_error_chi_square.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- Set the info for the plotting ----------

fig = plt.figure()
mpl.rcParams.update({'font.size': 30})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Right ascension (J2000)', fontsize=30)
plt.ylabel('Declination (J2000)', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Chi square')
fig.set_figheight(15)
fig.set_figwidth(15)

# ---------- Open the contour map (map1) ----------

Intensity_map1 = fits.open(FITS_file_5)
Coord1 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ----------- Create sub-images --------------------

cutout = Cutout2D(image_data1, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
plt.contour(hdu.data_c, levels=[-0.012*maxi, 0.012*maxi,
                                0.024*maxi, 0.048*maxi, 0.096*maxi, 0.192*maxi,
                                0.384*maxi, 0.768*maxi, 0.95*maxi], colors='black')
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------

plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map_chi_square.png',format='png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------
# -----------------------------------------------------------------
#                      Turnover intensity map
# -----------------------------------------------------------------
# -----------------------------------------------------------------


# ---------- Save the .fits for the turnover frequency map ----------

hdu.data = turn_over_fl
SM_filename = 'Turnover_flux_map.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- Set the info for the plotting ----------

fig = plt.figure()
mpl.rcParams.update({'font.size': 35})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Relative right ascension [mas]', fontsize=30)
plt.ylabel('Relative declination [mas]', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover flux [Jy/beam]')
fig.set_figheight(13)
fig.set_figwidth(13)

# ---------- Open the contour map (map5) ----------

Intensity_map1 = fits.open(FITS_file_5)
Coord1 = get_pkg_data_filename(FITS_file_5)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = [[0 for _ in range(y)] for _ in range(x)]

for i in range(0,y):
    for j in range(0,x):
        image_data1[i][j] = Intensity_map1[0].data[0][0][i][j]

# ----------- Create sub-images --------------------

cutout = Cutout2D(image_data1, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
max_xy = np.where(hdu.data_c == hdu.data_c.max() )
print(int(max_xy[0]))
print(int(max_xy[1]))
    
plt.contour(hdu.data_c, levels=[-0.012*maxi, 0.012*maxi,
                                0.024*maxi, 0.048*maxi, 0.096*maxi, 0.192*maxi,
                                0.384*maxi, 0.768*maxi, 0.95*maxi], colors='white',extent=[-17*0.16,(40-17)*0.16,-17*0.16,(40-17)*0.16])
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------

plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_flux_map.png',format='png', dpi=300, bbox_inches='tight')
plt.show()