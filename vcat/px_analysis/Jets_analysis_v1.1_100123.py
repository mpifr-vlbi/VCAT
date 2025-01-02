#--------------------------------------------------------------------
#--------------------------------------------------------------------

#                Code for analysing the jet profiles
#                          Developed by:
#                Luca Ricci (lricci@mpifr-bonn.mpg.de) 
    
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# IMPORTANT NOTE 1: Check "user input" to see where human interaction is required
# IMPORTANT NOTE 2: The fits map MUST be aligned along the Y axis, which the NORTHERN emission being
#                   the jet and the southern one being the counterjet.  
#                   If the jet (or counterjet) is bend, align along the Y axis the very first part of
#                   the emission, the code should be able to handle the further bend. To check, use
#                   the parameter finalmap_gaussian = -1.0 to check the analyzed slices.
# IMPORTANT NOTE 3: the code has been tested and works well for fairly align jets, bended jets need 
#                   to checked carefully (play with the parameter angle_for_slices to see which one
#                   fits best your source). The code doesn't work for double ridgline (in prep.)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st

#--------------------------------------------------------------------

#                          Commands 

#--------------------------------------------------------------------

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
from math import pi
from math import sqrt
from math import cos
from math import sin
from math import atan
from scipy import integrate
from scipy.interpolate import make_interp_spline, BSpline
fig,ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = [8,6]

#--------------------------------------------------------------------

#                         Inizialize arrays

#--------------------------------------------------------------------

X, Y = [], []
ratio = []
beta = []
open_anglej = []
open_anglej_err = []
open_angle = []
open_angle_err = []
widthj = []
widthj_err = []
width_err = []
distj = []
intensityj = []
Y_ridg = []
X_ridg = []
pos_x_ar = []

#------------------------ user input ---------------------------
pixel_size = 0.03           #in mas 
beam = 0.1916               #in mas 
fitsmap = 'Stacked_image_86GHz_rotated.fits'
max = 3.6                   # max pixel value (Jy/beam)
angle_for_slices = 0.0      # in degrees, decide the inclination of the slice <<<<------- extremely important value, source dependent
err_FWHM = 0.10             # indicative errors for a first fit of the collimation profile (in %)
err_flux = 0.20             # indicative errors for a first fit of the collimation profile (in %)
rms = 5.0e-3

#finalmap_gaussian = -1.0   # if -1.0 plot the final map + slices selected
finalmap_gaussian = 1.0     # if 1.0 plot the data + gaussian profile for #each slice

#counterjet = 1.0           # if counterjet = 1.0 the code will analyze the counterjet
counterjet = 0.0            # if counterjet = 0.0 the code will NOT analyze the counterjet

#info = 1.0                 # if info = 1.0 print the arrays of the different profiles
info = 0.0

fit = 1.0                   # if fit = 1.0 perform the fit on the collimation profile
#fit = 0.0
                
position_y = 512            # position (along y axis) of the core (i.e. brightest pixel OR modelfit core component)        
position_x = 511            # position (along x axis) of the core (i.e. brightest pixel OR modelfit core component)
position_x_beg = 490        # define the left boundary of the sub-map to analyze, must contain ALL the jet emission
position_x_fin = 530        # define the right boundary of the sub-map to analyze, must contain ALL the jet emission

#------------------------ user input ---------------------------

old_pos_a = 0
position_x_rg = position_x
position_y_rg = position_y
position_x_beg_rg = position_x_beg
position_x_fin_rg = position_x_fin

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                          Jet

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

position_y_beg = 513          # user input -- first y slice to analyze
position_y_fin = 565          # user input -- last y slice to analyze

if(counterjet==0.0):
    position_y_beg_rg = position_y_beg
position_y_fin_rg = position_y_fin

i = position_y_fin - position_y_beg
x = 0
cont = 0
a_old = 0
b_old = 0

for x in range(0, i):
    
    fig,ax = plt.subplots(1)

#------------------------ loading the map ---------------------------
    hdu = fits.open(fitsmap)[0]
#------------------------ loading the map ---------------------------
     
    position_y = position_y_beg + x
    print(position_y)
    Y_ridg.append(position_y)                     # y coordinate for the printing of the ridgeline
    size_x = position_x_fin - position_x_beg
    position_x = int(position_x_beg + size_x / 2)
    pos_x_ar.append(position_x)
    position = (position_x, position_y)
    size = (1, size_x)

#-------------------------------------------------------------------
#   define the ridgline, the perpendicular line and the pixels that will be analyzed
#--------------------------------------------------------------------

    cutout = Cutout2D(hdu.data, position, size)
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
    #print(data)
    print(ind)
    #print((position_x_beg + ind + 1.0) - position_x)
    #b = 0.0

#-------------------------------------------------------------------
#   conditions for a proper slice determination
#--------------------------------------------------------------------

    pos_a = position_x_beg + ind        # position of the maximum 
    X_ridg.append(pos_a)                # x coordinate for the printing of the ridgeline
    if(x == 0):                         # condition for the first slice
        old_pos_a = pos_a
        a = 0.0
        b = -a*angle_for_slices*pi/180.0
    if(x >= 1):
        a = pos_a - old_pos_a            #a is the key parameters: difference in the maximum between two consecutive slices
        b = -a*angle_for_slices*pi/180.0       #how much the slice should change angle
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

    if(finalmap_gaussian == -1.0 and b==0):
        plt.errorbar(x_line,y_line, color='yellow', markersize = 2.0, alpha = 0.1)
    if(finalmap_gaussian == -1.0 and b!=0):
        plt.errorbar(x_line,y_line, color='orange', markersize = 2.0, alpha = 0.8)

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

    data = []
    data_err = []
    for y in range(0, size_x):
        indx = x_line[y]
        #print(indx)
        indy = y_line_int[y]
        #print(indy)
        pix_val = hdu.data[indy,indx]
        if(pix_val >= 3.0*rms):   # very important value, it's the cutoff for each slice
            data.append(pix_val)
            data_err.append(pix_val*err_flux)

    #print(data)      #debugging stuff
    max_list = np.amax(data)
    size_x = len(data)


#--------------------------------------------------------------------
#                     Single gaussian fit
#--------------------------------------------------------------------

    X = np.linspace(1.0*pixel_size,size_x*pixel_size, size_x)
    if(finalmap_gaussian == 1.0):
        ax.errorbar(X, data, yerr = data_err, label='Data_point', color='red')
    model = models.Gaussian1D(max_list, size_x * pixel_size /2.0, beam)
    #model = models.Gaussian1D(max_list, size_x * pixel_size * 1.0/3.0, beam) + models.Gaussian1D(max_list, size_x * pixel_size * 2.0/3.0, beam)
    fitter = fitting.LevMarLSQFitter()
    #fitter = fitting.SLSQPLSQFitter()
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

    if(max_list <= 10.0*rms):   #very important value, it's the final cut, when the code should stop
        print('exit')
        break

    FWHM = 2.0*sqrt(2.0*np.log(2))*std
    #print('The FWHM is: ')
    #print(FWHM)
    chi_sq = 0.0
    for z in range(0, size_x):
        chi_sq +=  ( (data[z] - amplitude*np.exp( - (X[z] - mean)*(X[z] - mean) / (std*std*2.0)))**2.0 / (data_err[z]**2.0) )
    chi_sq_red = float(chi_sq / (size_x - 3))
    
    print('The chi_square_red is: ')
    print(chi_sq_red)
    if(chi_sq_red < 100.0): 
        if( (FWHM*FWHM - beam*beam) > 0.0):
            widthj.append( sqrt(FWHM*FWHM - beam*beam) )
            widthj_err.append(err_FWHM * sqrt(FWHM*FWHM - beam*beam))
            intensityj.append(a[0])
            cont += 1
            distj.append(cont*pixel_size)
            open_anglej.append( 2.0*atan( 0.5*sqrt(FWHM*FWHM - beam*beam) / (cont*pixel_size) ) * 180.0 / pi)
            open_anglej_err.append(err_FWHM*FWHM*4.0*cont*pixel_size*FWHM / (sqrt(FWHM*FWHM - beam*beam)*(4.0*cont*cont*pixel_size*pixel_size + FWHM*FWHM - beam*beam))) 
        if( (FWHM*FWHM - beam*beam) < 0.0):
            cont += 1
    if(chi_sq_red > 100.0):
        cont += 1

#--------------------------------------------------------------------
#                     plot data + gaussian
#--------------------------------------------------------------------

    if(finalmap_gaussian == 1.0):
        x = np.linspace(0,size_x*pixel_size,100)
        ax.errorbar(x, fitted_model(x), label='Gaussian_fit', color='blue')
        plt.ylabel('Jy/beam')
        plt.xlabel('mas')
        plt.legend(loc='upper right', ncol=1, handleheight = 2.0, labelspacing = 0.05, prop={'size': 14})
        plt.show()
        plt.close()

#--------------------------------------------------------------------
#                      Visualize the final map with all slices
#--------------------------------------------------------------------

if(finalmap_gaussian == -1.0):     
    print(X_ridg)     #debugging stuff
    print(Y_ridg)     #debugging stuff
    plt.plot(X_ridg, Y_ridg, color='red')
    Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.plasma)
    cbar = plt.colorbar(Intensity_map, fraction = 0.0477, pad = 0.00)
    cbar.ax.set_ylabel('mJy/beam')
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.contour(hdu.data, colors = 'white', alpha = 0.5, levels = [0.001*max, 0.002*max, 0.004*max, 0.008*max, 0.012*max, 0.034*max, 0.068*max, 0.126*max, 0.254*max])
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
    
    def func(x,a,b):
        return a*x**b
    
    popt, pcov = curve_fit(func, distj, widthj, sigma = widthj_err)
    perr = np.sqrt(np.diag(pcov))
    print('Merged fit:')
    print(popt)
    print(perr)

    xpoint = np.linspace(distj[0], distj[len(distj)-1], 1000)
    plt.plot(xpoint, func(xpoint,*popt), color='blue', label='')
    
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Jet width [mas]')
plt.xlabel('Distance [mas]')
plt.title('Collimation profile')
#plt.savefig('Collimation_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
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
plt.ylabel('Opening angle [deg]')
plt.xlabel('Distance [mas]')
plt.title('Opening angle')
#plt.savefig('Opening_angle_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()

#--------------------------------------------------------------------
#                      plot intensity
#--------------------------------------------------------------------

plt.close()

if(info == 1.0):
    print('Intensity data ------------------------------------------------------')
    print('X = ' + str(distj))
    print('Y = ' + str(intensityj))

plt.errorbar(distj,intensityj, fmt='o', markersize = 5.0)
plt.ylabel('Intensity [Jy/beam]')
plt.xlabel('Distance [mas]')
plt.title('Intensity Jet')
#plt.savefig('Intensity_profile_jet.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                          Counterjet

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------

#                         Inizialize arrays

#--------------------------------------------------------------------

X, Y = [], []
ratio = []
beta = []
open_anglecj = []
open_anglecj_err = []
open_anglec = []
open_angle_err = []
widthcj = []
widthcj_err = []
distcj = []
intensitycj = []
Y_ridg_c = []
X_ridg_c = []
pos_x_ar_cj = []


position_y_beg = 510   # user input -- first y slice to analyze
position_y_fin = 470   # user input -- last y slice to analyze

if(counterjet==1.0):
    position_y_beg_rg = position_y_fin

i = position_y_beg - position_y_fin
x = 0
cont = 0
a_old = 0
b_old = 0

if(counterjet == 1.0):
    for x in range(0, i):

#------------------------ loading the map ---------------------------
        hdu = fits.open(fitsmap)[0]
#------------------------ loading the map ---------------------------
             
        position_y = position_y_beg - x
        print(position_y)
        Y_ridg_c.append(position_y) # y coordinate for the printing of the ridgeline
        size_x = position_x_fin - position_x_beg
        position_x = int(position_x_beg + size_x / 2)
        pos_x_ar_cj.append(position_x)
        position = (position_x, position_y)
        size = (1, size_x)

#-------------------------------------------------------------------
#   define the ridgline, the perpendicular line and the pixels that will be analyzed
#--------------------------------------------------------------------

        cutout = Cutout2D(hdu.data, position, size)
        hdu.data2D = cutout.data

        x_line = []
        y_line = []
        y_line_int = []
        data = []
    
        for y in range(0, size_x):  #input from user
            pix_val = hdu.data2D[0,y]
            data.append(pix_val)     #save on an array the value of each pixel in the slice
            x_line.append(position_x_beg + y)   

        ind = np.argmax(data)
    #print(data)
        print(ind)
    #print((position_x_beg + ind + 1.0) - position_x)
    #b = 0.0

#-------------------------------------------------------------------
#   conditions for a proper slice determination
#--------------------------------------------------------------------

        pos_a = position_x_beg + ind # position of the maximum -- 6 should change accordingly to the souce (TEMPORARY)
        X_ridg_c.append(pos_a)               # x coordinate for the printing of the ridgeline
        if(x == 0):    # condition for the first slice
            old_pos_a = pos_a
            a = 0.0
            b = a*angle_for_slices*pi/180.0
        if(x >= 1):
            a = pos_a - old_pos_a  #a is the key parameters: difference in the maximum between two consecutive slices
            b = a*angle_for_slices*pi/180.0       #how much the slice should change angle
        if(a < a_old):         #if the variation is lower than before          
            diff = a_old - a   #how much is different
            b = b_old - diff*angle_for_slices*pi/180.0 #varying the angle accordingly
        if(a > a_old):         #if the variation is higher than before
            diff = a_old - a   #how much is different
            b = b_old - diff*angle_for_slices*pi/180.0 #varying the angle accordingly
        a_old = a
        old_pos_a = pos_a
        b_old = b
    
    
        q = position_y - sin(b)*(position_x_beg + ind)     #the line that will define the slice. q is the value when x = 0 --> needs to be the value of the most left point
        y_line = [q + sin(b)*z for z in x_line]    #y values for the slice
        y_line = np.array(y_line)
        y_line_int = y_line.astype(int)

        if(finalmap_gaussian == -1.0 and b==0):
            plt.errorbar(x_line,y_line, color='yellow', markersize = 2.0, alpha = 0.1)
        if(finalmap_gaussian == -1.0 and b!=0):
            plt.errorbar(x_line,y_line, color='orange', markersize = 2.0, alpha = 0.8)

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

        data = []
        for y in range(0, size_x):
            indx = x_line[y]
        #print(indx)
            indy = y_line_int[y]
        #print(indy)
            pix_val = hdu.data[indy,indx]
        #y += 1
            if(pix_val >= 3.0*rms):   # very important value, it's the cut for each slice
                data.append(pix_val)

    #print(data)      #debugging stuff
        max_list = np.amax(data)
        size_x = len(data)


#--------------------------------------------------------------------
#                     Gaussian fit
#--------------------------------------------------------------------

        X = np.linspace(1.0*pixel_size,size_x*pixel_size, size_x)
        if(finalmap_gaussian == 1.0):
            plt.plot(X, data, label='Data_point')
        model = models.Gaussian1D(max_list, size_x * pixel_size /2.0, beam)
        #model = models.Gaussian1D(max_list, size_x * pixel_size * 1.0/3.0, beam) + models.Gaussian1D(max_list, size_x * pixel_size * 2.0/3.0, beam)
        fitter = fitting.LevMarLSQFitter()
        #fitter = fitting.SLSQPLSQFitter()
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

        if(max_list <= 10.0*rms):   #very important value, it's the final cut, when the code should stop
            print('exit')
            break

        FWHM = 2.0*sqrt(2.0*np.log(2))*std
        #print('The FWHM is: ')
        #print(FWHM)
        chi_sq = 0.0
        for z in range(0, size_x):
            chi_sq +=  ( (data[z] - amplitude*np.exp( - (X[z] - mean)*(X[z] - mean) / (std*std*2.0)))**2.0 / (data[z]*data[z]*0.10*0.10) )
        chi_sq_red = float(chi_sq / (size_x - 3))
    
        print('The chi_square_red is: ')
        print(chi_sq_red)
        if(chi_sq_red < 100.0): 
            if( (FWHM*FWHM - beam*beam) > 0.0):
                widthcj.append( sqrt(FWHM*FWHM - beam*beam) )
                widthcj_err.append(err_FWHM * sqrt(FWHM*FWHM - beam*beam))
                intensitycj.append(a[0])
                cont += 1
                distcj.append(cont*pixel_size)
                open_anglecj.append( 2.0*atan( 0.5*sqrt(FWHM*FWHM - beam*beam) / (cont*pixel_size) ) * 180.0 / pi)
                open_anglecj_err.append(err_FWHM*FWHM*4.0*cont*pixel_size*FWHM / (sqrt(FWHM*FWHM - beam*beam)*(4.0*cont*cont*pixel_size*pixel_size + FWHM*FWHM - beam*beam))) 
            if( (FWHM*FWHM - beam*beam) < 0.0):
                cont += 1
        if(chi_sq_red > 100.0):
            cont += 1

#--------------------------------------------------------------------
#                     plot data + gaussian
#--------------------------------------------------------------------

        if(finalmap_gaussian == 1.0):
            x = np.linspace(0,size_x*pixel_size,100)
            plt.plot(x, fitted_model(x) , label='Gaussian_fit')
            plt.ylabel('Jy/beam')
            plt.xlabel('mas')
            plt.legend(loc='upper right')
            plt.show()

#--------------------------------------------------------------------
#                      Visualize the final map with all slices
#--------------------------------------------------------------------

    if(finalmap_gaussian == -1.0):     
        print(X_ridg)     #debugging stuff
        print(Y_ridg)     #debugging stuff
        plt.plot(X_ridg_c, Y_ridg_c, color='red')
        Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.plasma)
        cbar = plt.colorbar(Intensity_map, fraction = 0.0477, pad = 0.00)
        cbar.ax.set_ylabel('mJy/beam')
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.contour(hdu.data, colors = 'white', alpha = 0.5, levels = [0.001*max, 0.002*max, 0.004*max, 0.008*max, 0.012*max, 0.034*max, 0.068*max, 0.126*max, 0.254*max])
        plt.savefig('Maps_plus_slices_cj.png', format='png', dpi=300, bbox_inches = 'tight')
        plt.show()

#--------------------------------------------------------------------
#                      plot collimation
#--------------------------------------------------------------------

    plt.close()

    if(info == 1.0):
        print('Collimation data ------------------------------------------------------')
        print('X = [')
        print(distcj)
        print(']')
        print('Y = [')
        print(widthcj)
        print(']')
        print('Yerr = [')
        print(']')
        print(widthcj_err)

    plt.errorbar(distcj, widthcj, yerr = widthcj_err, fmt='o', markersize = 5.0)
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance [mas]')
    plt.title('Collimation profile')
    #plt.savefig('Collimation_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------
#                      plot opening angle
#--------------------------------------------------------------------

    plt.close()

    if(info == 1.0):
        print('Opening angle data ------------------------------------------------------')
        print('X = [')
        print(distcj)
        print('Y = [')
        print(']')
        print(open_anglecj)
        print(']')
        print('Yerr = [')
        print(open_anglecj_err)
        print(']')

    plt.errorbar(distcj,open_anglecj, yerr = open_anglecj_err, fmt='o', markersize = 5.0)
    plt.ylabel('Opening angle [deg]')
    plt.xlabel('Distance [mas]')
    plt.title('Opening angle')
    #plt.savefig('Opening_angle_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------
#                      plot intensity
#--------------------------------------------------------------------

    plt.close()

    if(info == 1.0):
        print('Intensity data ------------------------------------------------------')
        print('X = [')
        print(distcj)
        print(']')
        print('Y = [')
        print(intensitycj)
        print(']')

    plt.errorbar(distcj,intensitycj, fmt='o', markersize = 5.0)
    plt.ylabel('Intensity [Jy/beam]')
    plt.xlabel('Distance [mas]')
    plt.title('Intensity Jet')
    #plt.savefig('Intensity_profile_counterjet.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()
    
#--------------------------------------------------------------------
#                      Jet to counter-jet profile
#--------------------------------------------------------------------
    
    intensityj = np.array(intensityj)
    intensity = np.array(intensitycj)
    size = len(distcj)
    sep = []
    err_beta = []
    i = 0
    j = 0
    while i < size and j < size: 
        if(distj[i] == distcj[j]):
            ratio.append(intensityj[i]/intensity[j])
            sep.append(distj[i])
            i += 1
            j += 1
        elif(distj[i] > distcj[j]):
            j += 1
        elif(distj[i] < distcj[j]):
            i += 1
    
    print('Ratio ------------------------------------------------------')
    print('X = ')
    print(sep)
    print('Y = ')
    print(ratio)
    plt.errorbar(sep, ratio, fmt='o', markersize = 5.0)
    plt.ylabel('Ratio')
    plt.xlabel('Distance [mas]')
    #plt.savefig('Ratio_profile_.png', format='png', dpi=300, bbox_inches = 'tight')
    plt.show()

#--------------------------------------------------------------------

#                 Ridgeline
 
#--------------------------------------------------------------------

plt.close()

# -- create the subimage to print --

i = position_y_fin_rg - position_y_beg_rg
size_x = position_x_fin_rg - position_x_beg_rg  
position = (position_x_rg, position_y_rg + i/2)             #position of center, xposition of the map, yposition of the max plus half extension to center the map
b = 80       # b is aesthetic
size_y = i + b 
size = (size_y, size_x)
cutout = Cutout2D(hdu.data, position, size)
hdu.data2D = cutout.data

# -- define the ridgeline --

X_ridg = [(i - position_x_beg + (j - position_x_rg)) for i,j in zip(X_ridg,pos_x_ar)]
X_ridg_c = [(i - position_x_beg + (j - position_y_rg)) for i,j in zip(X_ridg_c, pos_x_ar_cj)]
l = len(X_ridg)
x = np.linspace(0,l,l)
Y_ridg = [b/2 + j for j in x]       
l_c = len(X_ridg_c)
x_c = np.linspace(0,l_c,l_c)
Y_ridg_c = [b/2 - j for j in x_c]

# -- print --

Intensity_map = plt.imshow(hdu.data2D, origin='lower', cmap=plt.cm.jet)
cbar = plt.colorbar(Intensity_map, fraction = 0.0477, pad = 0.00)
cbar.ax.set_ylabel('mJy/beam')
fig.set_figheight(10)
fig.set_figwidth(10)
plt.contour(hdu.data2D, colors = 'white', alpha = 0.5, levels = [0.001*max, 0.002*max, 0.004*max, 0.008*max, 0.012*max, 0.034*max, 0.068*max, 0.126*max, 0.254*max])
plt.plot(X_ridg,Y_ridg, color='red')
plt.plot(X_ridg_c,Y_ridg_c, color='red')
plt.savefig('Map_ridgeline.png', format='png', dpi=300, bbox_inches = 'tight')
plt.show()
