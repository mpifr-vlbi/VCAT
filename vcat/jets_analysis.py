#--------------------------------------------------------------------
#--------------------------------------------------------------------

#                Code for analysing the jet profiles
#                          Developed by:
#               Luca Ricci (luca.ricci.1195@gmail.com) 

# Version 1.0 does NOT include:

# - Double or multiple ridgeline
# - Properly handle very steep bends

#--------------------------------------------------------------------
#--------------------------------------------------------------------

# IMPORTANT NOTE 1: Check "user input" to see where human interaction may be required
# IMPORTANT NOTE 2: The fits map MUST be aligned along the Y axis, with the NORTHERN emission being
#                   the jet and the southern one being the counterjet.  
#                   If the jet (or counterjet) is bend, align along the Y axis the very first part of
#                   the emission, the code should be able to handle the further bend. To check, use
#                   the parameter finalmap_gaussian = -1.0 to check the analyzed slices.
# IMPORTANT NOTE 3: the code has been tested and works well for fairly align jets, bended jets need 
#                   to checked carefully (play with the parameter angle_for_slices to see which one
#                   fits best your source). The code doesn't work for double ridgline (in prep.)
# IMPORTANT NOTE 4: the first time the code is run for a new source do the rotation (option yes or only)
#                   or have the map you want to have analyzed named as "rotated.fits", since this is
#                   the input name that the code takes.

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------
#-------------------------------------------------------------------

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
# Function for reading input parameters
#-------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-fits", type=str, help="Filename of the FITS map")
    parser.add_argument("-px_size", type=float, help="Pixel size (in mas)")
    parser.add_argument("-beam", type=float, help="The beam of the map (in mas)")
    parser.add_argument("-angle", type=float, help="The slices angle")
    parser.add_argument("-rms", type=float, help="Rms noise (in Jy/beam)")
    parser.add_argument("-rotation", type=float, help="Pixel size (in mas/px)")

    args = parser.parse_args()
    return args

#--------------------------------------------------------------------

#                          Commands 

#--------------------------------------------------------------------

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
from math import pi
from math import sqrt
from math import sin
from math import atan
from scipy import integrate
fig,ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = [8,6]

#--------------------------------------------------------------------

#                         Inizialize arrays

#--------------------------------------------------------------------

X, Y            = [], []
ratio           = []
beta            = []
open_anglej     = []
open_anglej_err = []
open_angle      = []
open_angle_err  = []
widthj          = []
widthj_err      = []
width_err       = []
distj           = []
distj_int       = []
intensityj      = []
intensityj_err  = []
Y_ridg          = []
X_ridg          = []

#------------------------ parameters given by the terminal -------------------------------
args             = parse_arguments()
fitsmap          = args.fits
beam             = args.beam
pixel_size       = args.px_size
rms              = args.rms
angle_for_slices = args.angle
rotation_angle   = args.rotation

#----------------------- user input ----------------------
#finalmap_gaussian  = -1.0       # if -1.0 plot the final map + slices selected
finalmap_gaussian = 1.0        # if 1.0 plot the data + gaussian profile for #each slice

counterjet         = 1.0       # if counterjet = 1.0 the code will analyze the counterjet
#counterjet        = 0.0         # if counterjet = 0.0 the code will NOT analyze the counterjet

info              = 1.0         # if info = 1.0 print the arrays of the different profiles
#info               = 0.0

fit                = 1.0        # if fit = 1.0 perform the fit on the collimation profile
#fit               = 0.0

err_FWHM           = 0.10       # error to assume for the width obtained from the slicing/fitting (in %)
err_flux           = 0.05       # uncertainty on the brightness of each pixel  (in %)
err_flux_slice     = 0.10       # uncertainty on the integrated flux of each slice (in %)
mapsizex           = 1024       # map size (in pixel)
mapsizey           = 1024       # map size (in pixel)
j_len              = 100        # this parameters controls how many slices to analyse in the jet
cj_len             = 100        # this parameters controls how many slices to analyse in the counterjet
cut_radial         = 5.0        # *rms, it cut off the pixel below this level in each slice
cut_final          = 10.0       # *rms, it defines when to stop the code, if cut_final*rms higher than the brightest pixel in a slice, stop
chi_sq_val         = 100.0      # it's threshold for the gaussian fit. Try different ones
start_fit          = 5          # parameters that determine from which point onwards to perform the fit on the collimation profile (first method)
skip_fit           = 3          # parameters that determine how many points to skip before consider the next one for the fitting (first method)
avg_fit            = 3          # parameters that determine how many points to average for the fitting of the collimation profile (second method)
#------------------------ open CASA to rotate the image and save it --------------------------

response = input("Do you want to rotate the image? (yes/only/no, if only rotates the image and then stops the code): ")


if response.lower() == "yes":
    casa_executable = '/opt/casa-6.6.3-22-py3.8.el8/bin/casa' # user input: casapath needs to be written here
    if os.path.exists(casa_executable):
        # Construct the CASA script
        casa_script_rotation = f"""
importfits(fitsimage='{fitsmap}', imagename='imported.im', overwrite=True)
ia.open('imported.im')
ia.rotate(outfile = 'rotated.im', pa='{rotation_angle}deg', overwrite=True)
ia.close()
exportfits(imagename='rotated.im', fitsimage='rotated.fits', overwrite=True)
    """

        script_file = 'casa_script_rotation.py'
        with open(script_file, 'w') as f:
            f.write(casa_script_rotation)

        os.system(f'{casa_executable} --nolegger --nogui -c {script_file}')

    else:
        print('CASA executable not existing')

elif response.lower() == "only":

    casa_executable = '/home/lricci/Casa_versions/casa-6.6.0-20-py3.8.el7/bin/casa'
    if os.path.exists(casa_executable):
        # Construct the CASA script
        casa_script_rotation = f"""
importfits(fitsimage='{fitsmap}', imagename='imported.im', overwrite=True)
ia.open('imported.im')
ia.rotate(outfile = 'rotated.im', pa='{rotation_angle}deg', overwrite=True)
ia.close()
exportfits(imagename='rotated.im', fitsimage='rotated.fits', overwrite=True)
    """

        script_file = 'casa_script_rotation.py'
        with open(script_file, 'w') as f:
            f.write(casa_script_rotation)

        os.system(f'{casa_executable} --nolegger --nogui -c {script_file}')

    else:
        print('CASA executable not existing')

    exit()

elif response.lower() == "no":

    print('Proceeding with the code')

# Open rotated FITS file for the analyis
hdu = fits.open('rotated.fits')
image_data = hdu[0].data[0][0]
# ------------- Find the position of the maximum and create the box the analysis -------------
max_index = np.unravel_index(np.nanargmax(image_data, axis=None), image_data.shape)
max       = image_data[max_index[0],max_index[1]]
# --------------- user input -----------------
# IMPORTANT: if the maximum in the image DOES NOT correspond to the center (or position of the black hole)
# write in the position_y and position_x the actual desired starting position (use the rotate map for that).
position_y     = max_index[0]
position_y_or  = max_index[0]    # needed just for plotting
position_x     = max_index[1]
position_x_beg = position_x - 40
position_x_fin = position_x + 40
position_y_beg = position_y + 1      # plus 1 so the first slice considered is the first one of the jet (skip the core position)
position_y_fin = position_y + j_len   
position_y_cji = position_y - 1      # same thing as for the jet but for the counterjet
position_y_cjf = position_y - cj_len 
# --------------- user input ----------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                          Jet

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------ initializing some important parameters ------

if(counterjet==0.0):
    position_y_beg_rg = position_y_beg
position_y_fin_rg = position_y_fin
i = position_y_fin - position_y_beg
x = 0
cont = 0
a_old = 0
b_old = 0

for x in range(0, i):

# ------ initializing multiple plots if 1.0 ------
    if(finalmap_gaussian==1.0):
        fig,ax = plt.subplots(1)

# ------ initializing the different arrays ------

    position_y = position_y_beg + x
    print(position_y)
#   Y_ridg.append(position_y)                     
    size_x = position_x_fin - position_x_beg
    position_x = int(position_x_beg + size_x / 2)
    position = (position_x, position_y)
    size = (1, size_x)

#------------------------------------------------------------------------
#   create the array and the parameter (ind) for the slice determination
#--------------------------------------------------------------------

    cutout = Cutout2D(image_data, position, size)
    hdu.data2D = cutout.data

    x_line = []
    y_line = []
    y_line_int = []
    data = []
    
    for y in range(0, size_x):    
        pix_val = hdu.data2D[0,y]
        data.append(pix_val)               #save on an array the value of each pixel in the slice
        x_line.append(position_x_beg + y)   

    ind = np.argmax(data)
    #print(data)                                         # debugging stuff
    #print(ind)                                          # debugging stuff
    #print((position_x_beg + ind + 1.0) - position_x)    # debugging stuff


#-------------------------------------------------------------------
#   conditions for a proper slice determination
#--------------------------------------------------------------------

    pos_a = position_x_beg + ind                       # position of the maximum 
#    X_ridg.append(pos_a)                               # x coordinate for the printing of the ridgeline
    if(x == 0):                                        # condition for the first slice
        old_pos_a = pos_a
        a = 0.0
        b = -a*angle_for_slices*pi/180.0
    if(x >= 1):
        a = pos_a - old_pos_a                          #a is the key parameters: difference in the maximum between two consecutive slices
        b = -a*angle_for_slices*pi/180.0               #how much the slice should change angle
        if(a < a_old):                                 #if the variation is lower than before          
            diff = a_old - a                           #how much is different
            b = b_old + diff*angle_for_slices*pi/180.0 #varying the angle accordingly
        if(a > a_old):                                 #if the variation is higher than before
            diff = a_old - a                           #how much is different
            b = b_old + diff*angle_for_slices*pi/180.0 #varying the angle accordingly
        a_old = a
        old_pos_a = pos_a
        b_old = b
        
    q = position_y - sin(b)*(position_x_beg + ind)     #the line that will define the slice. q is the value when x = 0 --> needs to be the value of the most left point
    y_line = [q + sin(b)*z for z in x_line]            #y values for the slice
    y_line = np.array(y_line)
    y_line_int = y_line.astype(int)

#-------------------------------------------------------------------
#   debugging stuffs
#--------------------------------------------------------------------

    #print(x_line)
    #print(y_line_int)

#-------------------------------------------------------------------
#   fill out the array for gaussian analysis, check whether the slice is okay and then prepare for the output map
#--------------------------------------------------------------------

    indx = 0
    indy = 0

    data = []
    data_err = []
    for y in range(0, size_x):
        indx = x_line[y]
        #print(indx)
        indy = y_line_int[y]
        #print(indy)
        image_data = np.array(image_data)
        pix_val = image_data[indy,indx]
        if(pix_val >= cut_radial*rms):                
            data.append(pix_val)
            data_err.append(pix_val*err_flux)

    if(len(data) <= 5):
        print('Not this slice')
        if(finalmap_gaussian==1.0):
            plt.close()
        cont += 1
        continue

    max_list = np.amax(data)
    size_x = len(data)

    if(max_list <= cut_final*rms):
        print('Not this slice')
        if(finalmap_gaussian==1.0):
            plt.close()
        cont += 1
        continue


    if(finalmap_gaussian == -1.0 and b==0):
        plt.errorbar(x_line,y_line, color='yellow', markersize = 2.0, alpha = 0.5)
    if(finalmap_gaussian == -1.0 and b!=0):
        plt.errorbar(x_line,y_line, color='red', markersize = 2.0, alpha = 0.8)

    X_ridg.append(pos_a)
    Y_ridg.append(position_y)

#--------------------------------------------------------------------
#                     Single gaussian fit
#--------------------------------------------------------------------

    X = np.linspace(1.0*pixel_size,size_x*pixel_size, size_x)
    if(finalmap_gaussian == 1.0):
        ax.errorbar(X, data, yerr = data_err, label='Data_point', color='red')
    model = models.Gaussian1D(max_list, size_x * pixel_size /2.0, beam)
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(model, X, data)
    print(fitted_model)

#--------------------------------------------------------------------
#                     Gaussian integral
#--------------------------------------------------------------------

    amplitude = fitted_model.parameters[0]
    mean = fitted_model.parameters[1]
    std = fitted_model.parameters[2]

    x1 = 1.0*pixel_size
    x2 = size_x*pixel_size

    gauss = lambda x: amplitude*np.exp( - (x - mean)*(x - mean) / (std*std*2.0) )
    a = integrate.quad(gauss, x1, x2)
    #print('')
    #print('The flux of the slice is: ')
    #print(a)

    #if(max_list <= cut_final*rms):
    #    print('exit')
    #    break

    FWHM = 2.0*sqrt(2.0*np.log(2))*std
    print('The FWHM (convolved) is = ' + str(FWHM))
    chi_sq = 0.0
    for z in range(0, size_x):
        chi_sq +=  ( (data[z] - amplitude*np.exp( - (X[z] - mean)*(X[z] - mean) / (std*std*2.0)))**2.0 / (data_err[z]**2.0) )
    chi_sq_red = float(chi_sq / (size_x - 3))
    
    print('The chi_square_red is = ' + str(chi_sq_red))
    if(chi_sq_red < chi_sq_val):                 
        if( (FWHM*FWHM - beam*beam) > 0.0):
            widthj.append( sqrt(FWHM*FWHM - beam*beam) )
            print('The FWHM (de-convolved) is = ' + str( sqrt(FWHM**2.0 - beam**2.0) ))
            widthj_err.append(err_FWHM * sqrt(FWHM*FWHM - beam*beam))
            intensityj.append(a[0])
            cont += 1
            distj.append(cont*pixel_size)
            distj_int.append(cont*pixel_size)
            open_anglej.append( 2.0*atan( 0.5*sqrt(FWHM*FWHM - beam*beam) / (cont*pixel_size) ) * 180.0 / pi)
            open_anglej_err.append(err_FWHM*FWHM*4.0*cont*pixel_size*FWHM / (sqrt(FWHM*FWHM - beam*beam)*(4.0*cont*cont*pixel_size*pixel_size + FWHM*FWHM - beam*beam))) 
        if( (FWHM*FWHM - beam*beam) < 0.0):
            cont += 1
            distj_int.append(cont*pixel_size)
            intensityj.append(a[0])
    if(chi_sq_red > chi_sq_val):                 
        cont += 1

#--------------------------------------------------------------------
#                     plot data + gaussian
#--------------------------------------------------------------------

    if(finalmap_gaussian == 1.0):
        x = np.linspace(0,size_x*pixel_size,100)
        ax.errorbar(x, fitted_model(x), label='Gaussian_fit', color='blue')
        print(fitted_model(x))
        plt.ylabel('Jy/beam')
        plt.xlabel('mas')
        plt.legend(loc='upper right', ncol=1, handleheight = 2.0, labelspacing = 0.05, prop={'size': 14})
        plt.show()

#--------------------------------------------------------------------
#                      Visualize the final map with all slices
#--------------------------------------------------------------------

if(finalmap_gaussian == -1.0):     
    plt.plot(X_ridg, Y_ridg, color='red')
    Intensity_map = plt.imshow(image_data, origin='lower', cmap=plt.cm.plasma)
    plt.xlim(position_x-50,position_x+50)
    plt.ylim(position_y_or-90,position_y_or+200)
    cbar = plt.colorbar(Intensity_map, fraction = 0.0477, pad = 0.00)
    cbar.ax.set_ylabel('mJy/beam')
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.contour(image_data, colors = 'white', alpha = 0.5, levels = [3.0*rms, 6.0*rms, 12.0*rms, 24.0*rms, 48.0*rms, 96.0*rms, 192.*rms, 384.*rms])
    plt.savefig('Maps_plus_slices.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------
#                      plot collimation
#--------------------------------------------------------------------

plt.close()

if(info == 1.0):
    print('Collimation data ------------------------------------------------------')
    print('X = ' + str(distj))
    print('Y = ' + str(widthj))
    print('Yerr = ' + str(widthj_err))

plt.errorbar(distj, widthj, yerr = widthj_err, fmt='o', markersize = 5.0)

if(fit == 1.0):
    
    # -- Fitting function --
    def func(x,a,b):
        return a*x**b

    # -- Fitting arrays for: skip the first start_fit points, plus take one point every skip_fit --

    distj_fit      = distj[start_fit::skip_fit]            
    widthj_fit     = widthj[start_fit::skip_fit]         
    widthj_err_fit = widthj_err[start_fit::skip_fit]
    
    popt, pcov = curve_fit(func, distj_fit, widthj_fit, sigma = widthj_err_fit)
    perr = np.sqrt(np.diag(pcov))
    print('Fit values (a*x**b) with a the first term and b the second -- First method')
    print(popt)
    print(perr)

    plt.errorbar(distj_fit, widthj_fit, yerr= widthj_err_fit, fmt='o', color='red', markersize=7.0)
    xpoint = np.linspace(distj[0], distj[len(distj)-1], 1000)
    a = float(popt[0])
    b = float(popt[1])
    plt.text(xpoint[1], widthj[len(widthj)-2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12, bbox=dict(facecolor='red', alpha = 0.5))
    plt.plot(xpoint, func(xpoint, *popt), color='red')
    
    # -- Fitting arrays for: take an average value every avg_fit points --

    distj_fit       = []
    widthj_fit      = []
    widthj_err_fit  = []

    counter = 0
    valuer  = 0.0
    valued  = 0.0
    valuee  = 0.0
    
    for i in range(0,len(distj)):
        counter = counter + 1
        if(counter <= avg_fit):
            valuer = valuer + distj[i]
            valued = valued + widthj[i]
            valuee = valuee + widthj_err[i]
            
        if(counter == avg_fit+1):
            valuer = valuer / float(avg_fit)         
            valued = valued / float(avg_fit)
            valuee = valuee / float(avg_fit)

            # Fill out the array for the fitting
            distj_fit.append(valuer)
            widthj_fit.append(valued)
            widthj_err_fit.append(valuee)
             

            # Reset values 
            counter = 1
            valuer  = 0.0
            valued  = 0.0
            valuee  = 0.0

            valuer = valuer + distj[i]
            valued = valued + widthj[i]
            valuee = valuee + widthj_err[i]

    popt, pcov = curve_fit(func, distj_fit, widthj_fit, sigma = widthj_err_fit)
    perr = np.sqrt(np.diag(pcov))
    print('Fit values (a*x**b) with a the first term and b the second -- Second method')
    print(popt)
    print(perr)

    print('Valori fit media')
    print(distj_fit)
    print(widthj_fit)
    plt.errorbar(distj_fit, widthj_fit, yerr= widthj_err_fit, fmt='o', color='purple', markersize=7.0)
    a = float(popt[0])
    b = float(popt[1])
    plt.text(xpoint[80], widthj[len(widthj)-2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12, bbox=dict(facecolor='purple', alpha=0.5))    
    plt.plot(xpoint, func(xpoint, *popt), color='purple')

plt.xscale('log')
plt.yscale('log')
plt.ylabel('Jet width [mas]')
plt.xlabel('Distance [mas]')
plt.title('Collimation profile')
plt.savefig('Collimation_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------
#                      plot opening angle
#--------------------------------------------------------------------

plt.close()

if(info == 1.0):
    print('Opening angle data ------------------------------------------------------')
    print('X = ' + str(distj))
    print('Y = ' + str(open_anglej))
    print('Yerr = ' + str(open_anglej_err))

plt.errorbar(distj,open_anglej, yerr = open_anglej_err, fmt='o', markersize = 5.0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Opening angle [deg]')
plt.xlabel('Distance [mas]')
plt.title('Opening angle')
plt.savefig('Opening_angle_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------
#                      plot intensity
#--------------------------------------------------------------------

plt.close()

for i in range(0,len(intensityj)):
    intensityj_err.append(intensityj[i]*err_flux_slice)

if(info == 1.0):
    print('Intensity data ------------------------------------------------------')
    print('X = ' + str(distj_int))
    print('Y = ' + str(intensityj))
    print('Yerr = ' + str(intensityj_err))

plt.errorbar(distj_int,intensityj, yerr=intensityj_err, fmt='o', markersize = 5.0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Intensity [Jy/beam]')
plt.xlabel('Distance [mas]')
plt.title('Intensity Jet')
plt.savefig('Intensity_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                          Counterjet

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------

#                         Inizialize arrays

#--------------------------------------------------------------------

X, Y             = [], []
ratio            = []
beta             = []
open_anglecj     = []
open_anglecj_err = []
open_anglec      = []
open_angle_err   = []
widthcj          = []
widthcj_err      = []
distcj           = []
distcj_int       = []
intensitycj      = []
intensitycj_err  = []
Y_ridg_c         = []
X_ridg_c         = []

position_y_beg = position_y_cji
position_y_fin = position_y_cjf

if(counterjet==1.0):
    position_y_beg_rg = position_y_fin

i = position_y_beg - position_y_fin
x = 0
cont = 0
a_old = 0
b_old = 0

if(counterjet == 1.0):
    for x in range(0, i):
            
        # ------ initializing multiple plots if 1.0 ------
        if(finalmap_gaussian==1.0):
            fig,ax = plt.subplots(1)

        position_y = position_y_beg - x
        print(position_y)
#        Y_ridg_c.append(position_y) # y coordinate for the printing of the ridgeline
        size_x = position_x_fin - position_x_beg
        position_x = int(position_x_beg + size_x / 2)
        position = (position_x, position_y)
        size = (1, size_x)

#-------------------------------------------------------------------
#   create the array and the parameter (ind) for the slice determination
#--------------------------------------------------------------------

        cutout = Cutout2D(image_data, position, size)
        hdu.data2D = cutout.data

        x_line = []
        y_line = []
        y_line_int = []
        data = []
    
        for y in range(0, size_x): 
            pix_val = hdu.data2D[0,y]
            data.append(pix_val)                  #save on an array the value of each pixel in the slice
            x_line.append(position_x_beg + y)   

        ind = np.argmax(data)
        #print(data)
        #print(ind)
        #print((position_x_beg + ind + 1.0) - position_x)
        #b = 0.0

#-------------------------------------------------------------------
#   conditions for a proper slice determination
#--------------------------------------------------------------------

        pos_a = position_x_beg + ind         # position of the maximum -- 6 should change accordingly to the souce (TEMPORARY)
#        X_ridg_c.append(pos_a)               # x coordinate for the printing of the ridgeline
        if(x == 0):                          # condition for the first slice
            old_pos_a = pos_a
            a = 0.0
            b = a*angle_for_slices*pi/180.0
        if(x >= 1):
            a = pos_a - old_pos_a                       #a is the key parameters: difference in the maximum between two consecutive slices
            b = a*angle_for_slices*pi/180.0             #how much the slice should change angle
        if(a < a_old):                                  #if the variation is lower than before          
            diff = a_old - a                            #how much is different
            b = b_old - diff*angle_for_slices*pi/180.0  #varying the angle accordingly
        if(a > a_old):                                  #if the variation is higher than before
            diff = a_old - a                            #how much is different
            b = b_old - diff*angle_for_slices*pi/180.0  #varying the angle accordingly
        a_old = a
        old_pos_a = pos_a
        b_old = b
    
    
        q = position_y - sin(b)*(position_x_beg + ind)  #the line that will define the slice. q is the value when x = 0 --> needs to be the value of the most left point
        y_line = [q + sin(b)*z for z in x_line]         #y values for the slice
        y_line = np.array(y_line)
        y_line_int = y_line.astype(int)

#        if(finalmap_gaussian == -1.0 and b==0):
#            plt.errorbar(x_line,y_line, color='yellow', markersize = 2.0, alpha = 0.5)
#        if(finalmap_gaussian == -1.0 and b!=0):
#            plt.errorbar(x_line,y_line, color='red', markersize = 2.0, alpha = 0.8)

#-------------------------------------------------------------------
#   debugging stuffs
#--------------------------------------------------------------------

        #print(x_line)
        #print(y_line_int)

#-------------------------------------------------------------------
#   fill out the array for gaussian analysis
#--------------------------------------------------------------------

        indx = 0
        indy = 0

        data     = []
        data_err = []
        for y in range(0, size_x):
            indx = x_line[y]
            #print(indx)
            indy = y_line_int[y]
            #print(indy)
            image_data = np.array(image_data)
            pix_val = image_data[indy,indx]
            #y += 1
            if(pix_val >= cut_radial*rms):       
                data.append(pix_val)
                data_err.append(pix_val*err_flux)

        #print(data)      #debugging stuff
 #       max_list = np.amax(data)
 #       size_x = len(data)

        if(len(data) <= 5):
            print('Not this slice')
            if(finalmap_gaussian==1.0):
                plt.close()
            cont += 1
            continue

        max_list = np.amax(data)
        size_x = len(data)

        if(max_list <= cut_final*rms):
            print('Not this slice')
            if(finalmap_gaussian==1.0):
                plt.close()
            cont += 1
            continue


        if(finalmap_gaussian == -1.0 and b==0):
            plt.errorbar(x_line,y_line, color='yellow', markersize = 2.0, alpha = 0.5)
        if(finalmap_gaussian == -1.0 and b!=0):
            plt.errorbar(x_line,y_line, color='red', markersize = 2.0, alpha = 0.8)

        X_ridg_c.append(pos_a)
        Y_ridg_c.append(position_y)


#--------------------------------------------------------------------
#                     Gaussian fit
#--------------------------------------------------------------------

        X = np.linspace(1.0*pixel_size,size_x*pixel_size, size_x)
        if(finalmap_gaussian == 1.0):
            ax.errorbar(X, data, yerr = data_err, label='Data_point', color='red')
        model = models.Gaussian1D(max_list, size_x * pixel_size /2.0, beam)
        fitter = fitting.LevMarLSQFitter()
        fitted_model = fitter(model, X, data)
        print(fitted_model)

#--------------------------------------------------------------------
#                     Gaussian integral
#--------------------------------------------------------------------

        amplitude = fitted_model.parameters[0]
        mean = fitted_model.parameters[1]
        std = fitted_model.parameters[2]

        x1 = 1.0*pixel_size
        x2 = size_x*pixel_size

        gauss = lambda x: amplitude*np.exp( - (x - mean)*(x - mean) / (std*std*2.0) )
        a = integrate.quad(gauss, x1, x2)
        #print('')
        #print('The flux of the slice is: ')
        #print(a)

        if(max_list <= cut_final*rms):          
            print('exit')
            continue

        FWHM = 2.0*sqrt(2.0*np.log(2))*std
        print('The FWHM (convolved) is = ' + str(FWHM))
        #print('The FWHM is: ')
        #print(FWHM)
        chi_sq = 0.0
        for z in range(0, size_x):
            chi_sq +=  ( (data[z] - amplitude*np.exp( - (X[z] - mean)*(X[z] - mean) / (std*std*2.0)))**2.0 / (data[z]*data[z]*0.10*0.10) )
        chi_sq_red = float(chi_sq / (size_x - 3))
    
        print('The chi_square_red is: ')
        print(chi_sq_red)
        if(chi_sq_red < chi_sq_val): 
            if( (FWHM*FWHM - beam*beam) > 0.0):
                widthcj.append( sqrt(FWHM*FWHM - beam*beam) )
                print('The FWHM (de-convolved) is = ' + str( sqrt(FWHM**2.0 - beam**2.0) ))
                widthcj_err.append(err_FWHM * sqrt(FWHM*FWHM - beam*beam))
                intensitycj.append(a[0])
                cont += 1
                distcj.append(cont*pixel_size)
                distcj_int.append(cont*pixel_size)
                open_anglecj.append( 2.0*atan( 0.5*sqrt(FWHM*FWHM - beam*beam) / (cont*pixel_size) ) * 180.0 / pi)
                open_anglecj_err.append(err_FWHM*FWHM*4.0*cont*pixel_size*FWHM / (sqrt(FWHM*FWHM - beam*beam)*(4.0*cont*cont*pixel_size*pixel_size + FWHM*FWHM - beam*beam))) 
            if( (FWHM*FWHM - beam*beam) < 0.0):
                cont += 1
                distcj_int.append(cont*pixel_size)
                intensitycj.append(a[0])
        if(chi_sq_red > chi_sq_val):
            cont += 1

#--------------------------------------------------------------------
#                     plot data + gaussian
#--------------------------------------------------------------------

        if(finalmap_gaussian == 1.0):
            x = np.linspace(0,size_x*pixel_size,100)
            ax.errorbar(x, fitted_model(x) , label='Gaussian_fit', color='blue')
            plt.ylabel('Jy/beam')
            plt.xlabel('mas')
            plt.legend(loc='upper right', ncol=1, handleheight = 2.0, labelspacing = 0.05, prop={'size': 14})
            plt.show()

#--------------------------------------------------------------------
#                      Visualize the final map with all slices
#--------------------------------------------------------------------

    if(finalmap_gaussian == -1.0):     
        #print(X_ridg)     #debugging stuff
        #print(Y_ridg)     #debugging stuff
        plt.plot(X_ridg_c, Y_ridg_c, color='red')
        Intensity_map = plt.imshow(image_data, origin='lower', cmap=plt.cm.plasma)
        cbar = plt.colorbar(Intensity_map, fraction = 0.0477, pad = 0.00)
        plt.xlim(position_x-50,position_x+50)
        plt.ylim(position_y_or-90,position_y_or+100)
        cbar.ax.set_ylabel('mJy/beam')
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.contour(image_data, colors = 'white', alpha = 0.5, levels = [3.0*rms, 6.0*rms, 12.0*rms, 24.0*rms, 48.0*rms, 96.0*rms, 192.*rms, 384.*rms])
        plt.savefig('Maps_plus_slices_cj.png', format='png', dpi=300, bbox_inches = 'tight')
        plt.show()

#--------------------------------------------------------------------
#                      plot collimation
#--------------------------------------------------------------------

    plt.close()

    if(info == 1.0):
        print('Collimation data ------------------------------------------------------')
        print('X = ' + str(distcj))
        print('Y = ' + str(widthcj))
        print('Yerr = ' + str(widthcj_err))

    if(fit == 1.0):
    
        # -- Fitting function --
        def func(x,a,b):
            return a*x**b

       # -- Fitting arrays for: skip the first start_fit points, plus take one point every skip_fit --

        distcj_fit      = distcj[start_fit::skip_fit]           
        widthcj_fit     = widthcj[start_fit::skip_fit]         
        widthcj_err_fit = widthcj_err[start_fit::skip_fit]
    
        popt, pcov = curve_fit(func, distcj_fit, widthcj_fit, sigma = widthcj_err_fit)
        perr = np.sqrt(np.diag(pcov))
        print('Fit values (a*x**b) with a the first term and b the second -- First method')
        print(popt)
        print(perr)

        plt.errorbar(distcj_fit, widthcj_fit, yerr= widthcj_err_fit, fmt='o', color='red', markersize=7.0)
        xpoint = np.linspace(distcj[0], distcj[len(distcj)-1], 1000)
        a = float(popt[0])
        b = float(popt[1])
        plt.text(xpoint[1], widthcj[len(widthcj)-2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12, bbox=dict(facecolor='red', alpha = 0.5))
        plt.plot(xpoint, func(xpoint, *popt), color='red')
    
       # -- Fitting arrays for: take an average value every avg_fit points --

        distcj_fit       = []
        widthcj_fit      = []
        widthcj_err_fit  = []

        counter = 0
        valuer  = 0.0
        valued  = 0.0
        valuee  = 0.0

        for i in range(0,len(distcj)):
            counter = counter + 1
            if(counter <= avg_fit):
                valuer = valuer + distcj[i]
                valued = valued + widthcj[i]
                valuee = valuee + widthcj_err[i]
            
            if(counter == avg_fit+1):
                valuer = valuer / float(avg_fit)         
                valued = valued / float(avg_fit)
                valuee = valuee / float(avg_fit)

                # Fill out the array for the fitting
                distcj_fit.append(valuer)
                widthcj_fit.append(valued)
                widthcj_err_fit.append(valuee)
             

                # Reset values 
                counter = 1
                valuer  = 0.0
                valued  = 0.0
                valuee  = 0.0

                valuer = valuer + distcj[i]
                valued = valued + widthcj[i]
                valuee = valuee + widthcj_err[i]

        popt, pcov = curve_fit(func, distcj_fit, widthcj_fit, sigma = widthcj_err_fit)
        perr = np.sqrt(np.diag(pcov))
        print('Fit values (a*x**b) with a the first term and b the second -- Second method')
        print(popt)
        print(perr)
 
    print('Valori fit media')
    print(distcj_fit)
    print(widthcj_fit)
    plt.errorbar(distcj_fit, widthcj_fit, yerr= widthcj_err_fit, fmt='o', color='purple', markersize=7.0)
    a = float(popt[0])
    b = float(popt[1])
    plt.text(xpoint[300], widthcj[len(widthcj)-2], f'$y = {a:.2f} \cdot x^{{{b:.2f}}}$', fontsize=12, bbox=dict(facecolor='purple', alpha=0.5))    
    plt.plot(xpoint, func(xpoint, *popt), color='purple')


    plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(distcj, widthcj, yerr = widthcj_err, fmt='o', markersize = 5.0)
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance [mas]')
    plt.title('Collimation profile')
    plt.savefig('Collimation_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------
#                      plot opening angle
#--------------------------------------------------------------------

    plt.close()

    if(info == 1.0):
        print('Opening angle data ------------------------------------------------------')
        print('X = ' + str(distcj))
        print('Y = ' + str(open_anglecj))
        print('Yerr = ' + str(open_anglecj_err))

    plt.errorbar(distcj,open_anglecj, yerr = open_anglecj_err, fmt='o', markersize = 5.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Opening angle [deg]')
    plt.xlabel('Distance [mas]')
    plt.title('Opening angle')
    plt.savefig('Opening_angle_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------
#                      plot intensity
#--------------------------------------------------------------------

    plt.close()

    for i in range(0,len(intensitycj)):
        intensitycj_err.append(intensitycj[i]*err_flux_slice)

    if(info == 1.0):
        print('Intensity data ------------------------------------------------------')
        print('X = ' + str(distcj_int))
        print('Y = ' + str(intensitycj))
        print('Yerr = ' + str(intensitycj_err))

    plt.errorbar(distcj_int,intensitycj, yerr=intensitycj_err, fmt='o', markersize = 5.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Intensity [Jy/beam]')
    plt.xlabel('Distance [mas]')
    plt.title('Intensity Jet')
    plt.savefig('Intensity_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()
    
#--------------------------------------------------------------------
#                      Jet to counter-jet profile
#--------------------------------------------------------------------
    
    sep      = []
    err_beta = []

    intensityj = np.array(intensityj)
    intensity = np.array(intensitycj)
    size = min(len(distj_int),len(distcj_int))

    i = 0
    j = 0
    while i < size and j < size: 
        if(distj_int[i] == distcj_int[j]):
            ratio.append(intensityj[i]/intensity[j])
            err_beta.append( sqrt( (1.0/intensity[i])**2.0 * intensityj_err[i]**2.0 + 
                                  (intensityj[i]/(intensity[i]**2.0))**2.0 * intensitycj_err[i]**2.0 ) )
            sep.append(distj_int[i])
            i += 1
            j += 1
        elif(distj_int[i] > distcj_int[j]):
            j += 1
        elif(distj_int[i] < distcj_int[j]):
            i += 1

    if(info==1):
        print('Ratio ------------------------------------------------------')
        print('X = ' + str(sep))
        print('Y = ' + str(ratio))
        print('Yerr = ' + str(err_beta))

    plt.errorbar(sep, ratio, yerr=err_beta, fmt='o', markersize = 5.0)
    plt.ylabel('Ratio')
    plt.xlabel('Distance [mas]')
    plt.savefig('Ratio_profile_.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()
