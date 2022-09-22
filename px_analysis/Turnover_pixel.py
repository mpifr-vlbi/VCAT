#Code to read FITS file given in input

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import log as ln
from scipy.optimize import curve_fit

#Libraries for reading and manipulate .fits files

from astropy.io import fits
from math import sqrt
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import Cutout2D

fig, ax = plt.subplots(1)
plt.rcParams["figure.figsize"] = [8,6]
#Turnover frequency function
def func_turn(x,I0,turn,alpha0):
    alphat = 2.5
    return I0 * (x/turn)**alphat * (1.0 - np.exp( - (turn/x)**(alphat - alpha0)) )

#Name of the fits file you want to analyze

FITS_file_1 = 'NGC315_5_2020_coreturn_cut.fits'
nu_1 = 5.         #frequency
rms_1 = 2.0e-5    #rms noise
err_1 = 0.10      #error on the brightness

FITS_file_2 = 'NGC315_8_2020_coreturn_cut.fits'
nu_2 = 8.         #frequency
rms_2 = 7.0e-5    #rms noise
err_2 = 0.10      #error on the brightness

FITS_file_3 = 'NGC315_15_2020_coreturn_cut.fits'
nu_3 = 15.        #frequency
rms_3 = 6.0e-5    #rms noise
err_3 = 0.10      #error on the brightness

FITS_file_4 = 'NGC315_22_2020_coreturn_cut.fits'
nu_4 = 22.        #frequency
rms_4 = 5.0e-5    #rms noise
err_4 = 0.10      #error on the brightness


fr = [nu_1, nu_2, nu_3, nu_4]   #array of frequencies
thres = 10.0                    #threshold for rms cut

#---------------------------------------------------------------------------------
#                                 Turnover map
#---------------------------------------------------------------------------------

# ---------- Open the .fits file number 1 ----------

Intensity_map1 = fits.open(FITS_file_1)
Coord1 = get_pkg_data_filename(FITS_file_1)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data1 = Intensity_map1[0].data

#If you want to know the dimension of the array
print(type(image_data1))
print(image_data1.shape)

# ---------- Open the .fits file number 2 ----------

Intensity_map2 = fits.open(FITS_file_2)
Coord2 = get_pkg_data_filename(FITS_file_2)
hdu = fits.open(Coord2)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map2.info()

#Save the data in a numpy array
image_data2 = Intensity_map2[0].data

#If you want to know the dimension of the array
print(type(image_data2))
print(image_data2.shape)

# ---------- Open the .fits file number 3 ----------

Intensity_map3 = fits.open(FITS_file_3)
Coord3 = get_pkg_data_filename(FITS_file_3)
hdu = fits.open(Coord3)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map3.info()

#Save the data in a numpy array
image_data3 = Intensity_map3[0].data

#If you want to know the dimension of the array
print(type(image_data3))
print(image_data3.shape)

# ---------- Open the .fits file number 4 ----------

Intensity_map4 = fits.open(FITS_file_4)
Coord4 = get_pkg_data_filename(FITS_file_4)
hdu = fits.open(Coord4)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map4.info()

#Save the data in a numpy array
image_data4 = Intensity_map4[0].data

#If you want to know the dimension of the array
print(type(image_data4))
print(image_data4.shape)

x = image_data1.shape[0]
y = image_data2.shape[1]

#Initialize arrays

turnover = [[0 for _ in range(x)] for _ in range(y)]
err_turn = [[0 for _ in range(x)] for _ in range(y)]
chi_square = [[0 for _ in range(x)] for _ in range(y)]

print('---------------------------------------------------------')

for j in range(0,y):
    for i in range(0,x):
        
        bright = []
        err_bright = []
        #fill in arrays
        
        if(image_data1[i][j] > thres*rms_1):
            bright.append(image_data1[i][j])
            err_bright.append(image_data1[i][j]*err_1)
        if(image_data2[i][j] > thres*rms_2):
            bright.append(image_data2[i][j])
            err_bright.append(image_data2[i][j]*err_2)
        if(image_data3[i][j] > thres*rms_3):        
            bright.append(image_data3[i][j])
            err_bright.append(image_data3[i][j]*err_3)
        if(image_data4[i][j] > thres*rms_4):
            bright.append(image_data4[i][j])
            err_bright.append(image_data4[i][j]*err_4)
        
        #analyze only pixels in which all the maps have points 
        
        if(len(bright) == 4):
            popt, pcov = curve_fit(func_turn, fr, bright, sigma = err_bright, maxfev = 1000000)
            perr = np.sqrt(np.diag(pcov))
            chi_sq = 0.0
            for p in range(0,len(bright)):
                chi_sq = chi_sq + (bright[p] - func_turn(fr[p],*popt))**2.0 / (err_bright[p]**2.0)
            chi_square[i][j] = chi_sq
            
            #use this to check some specific pixels
            
            if(i == 512 and j == 512):
                x_p = np.linspace(5,25,1000)
                ax.errorbar(fr, bright, yerr=err_bright, fmt='o')
                plt.plot(x_p,func_turn(x_p,*popt), label='Best fit')
                plt.plot(x_p,func_turn(x_p,popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2]), label='Best fit +err')
                plt.plot(x_p,func_turn(x_p,popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2]), label='Best fit -err')
                axes = plt.gca()
                axes.xaxis.label.set_size(16)
                axes.yaxis.label.set_size(16)
                ax.tick_params(axis='x', labelsize=20)
                ax.tick_params(axis='y', labelsize=20)
                plt.xlabel('frequency [GHz]')
                plt.ylabel('Brightness [Jy/beam]')
                plt.legend(loc='upper left', ncol=1, handleheight = 1.0, labelspacing=0.005, prop={'size':16})
                filename = 'profile.png'
                plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                
            # fill in the final array
            
            turnover[i][j] = popt[1]
            err_turn[i][j] = perr[1]
            
            #if turnover outside of the frequency range set to zero
            
            if(popt[1] > 18.0 or popt[1] < 5.0):
                turnover[i][j] = 0.0
                err_turn[i][j] = 0.0 
                chi_square[i][j] = 0.0
        if(len(bright) != 4):
            turnover[i][j] = 0.0
            err_turn[i][j] = 0.0
            chi_square[i][j] = 0.0
            
position = (515, 515)
size = (40,40)
        
# -----------------------------------------------------------------
# ---------- Turnover frequency map ----------
# -----------------------------------------------------------------

# ---------- Save the .fits for the turnover map ----------

hdu.data = turnover
SM_filename = 'Turnover_map.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- TM map ----------

fig = plt.figure()
fig.add_subplot(111, projection=wcs)
fig.add_subplot(projection=wcs)
mpl.rcParams.update({'font.size': 30})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet, vmax = nu_4, vmin = nu_1)
plt.xlabel('Right ascension (J2000)', fontsize=30)
plt.ylabel('Declination (J2000)', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover frequency')
fig.set_figheight(15)
fig.set_figwidth(15)


# ---------- Open the contour map ----------

FITS_file = 'NGC315_22_2020_coreturn_cut.fits'

Intensity_map1 = fits.open(FITS_file)
Coord1 = get_pkg_data_filename(FITS_file)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data_c = Intensity_map1[0].data

# ---------- Save the .fits for the spectral index map ----------

hdu.data_c = image_data_c.data

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data_c = cutout.data
turnover_cut = Cutout2D(turnover, position, size)
plt.contour(hdu.data_c, levels=[-0.003*0.327, 0.003*0.327, 0.006*0.327, 0.012*0.327, 0.024*0.327, 0.048*0.327, 0.096*0.327, 
                                0.192*0.327, 0.348*0.327, 0.768*0.327], colors='black')
#plt.contour(turnover_cut.data, levels=[5, 7.5, 10., 12.5, 15, 17.5], colors='red')

# ----------------- Visualize map --------------
plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------
# ---------- Error map ----------
# -----------------------------------------------------------------

# ---------- Save the .fits for the spectral index map ----------

hdu.data = err_turn
SM_filename = 'Turnover_map_error.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- SM map ----------

fig = plt.figure()
fig.add_subplot(111, projection=wcs)
fig.add_subplot(projection=wcs)
mpl.rcParams.update({'font.size': 30})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Right ascension (J2000)', fontsize=30)
plt.ylabel('Declination (J2000)', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Turnover frequency error')
fig.set_figheight(15)
fig.set_figwidth(15)


# ---------- Open the contour map ----------

FITS_file = 'NGC315_22_2020_coreturn_cut.fits'

Intensity_map1 = fits.open(FITS_file)
Coord1 = get_pkg_data_filename(FITS_file)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data_c = Intensity_map1[0].data

# ---------- Save the .fits for the spectral index map ----------

hdu.data_c = image_data_c.data

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data_c = cutout.data
plt.contour(hdu.data_c, levels=[-0.003*0.327, 0.003*0.327, 0.006*0.327, 0.012*0.327, 0.024*0.327, 0.048*0.327, 0.096*0.327, 
                                0.192*0.327, 0.348*0.327, 0.768*0.327], colors='black')

# ----------------- Visualize map --------------
plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Turnover_map_error.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------
# ---------- Chi_square_map ----------
# -----------------------------------------------------------------


# ---------- Save the .fits for the spectral index map ----------

hdu.data = chi_square
SM_filename = 'Chi_square_map.fits'
hdu.writeto(SM_filename, overwrite=True)

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data = cutout.data
#hdu.header.update(cutout.wcs.to_header())

# ---------- SM map ----------

fig = plt.figure()
fig.add_subplot(111, projection=wcs)
fig.add_subplot(projection=wcs)
mpl.rcParams.update({'font.size': 30})
plt.tick_params(axis='both', labelsize = 35)
Intensity_map = plt.imshow(hdu.data, origin='lower', cmap=plt.cm.jet)
plt.xlabel('Right ascension (J2000)', fontsize=30)
plt.ylabel('Declination (J2000)', labelpad=-1.0)
cbar = plt.colorbar(Intensity_map, fraction=0.0475, pad=0.00)
cbar.ax.set_ylabel('Chi square')
fig.set_figheight(15)
fig.set_figwidth(15)


# ---------- Open the contour map ----------

FITS_file = 'NGC315_22_2020_coreturn_cut.fits'

Intensity_map1 = fits.open(FITS_file)
Coord1 = get_pkg_data_filename(FITS_file)
hdu = fits.open(Coord1)[0]
wcs = WCS(hdu.header)

#Information about the file you opened
Intensity_map1.info()

#Save the data in a numpy array
image_data_c = Intensity_map1[0].data

# ---------- Save the .fits for the spectral index map ----------

hdu.data_c = image_data_c.data

# ----------- Create sub-images --------------------

cutout = Cutout2D(hdu.data, position, size)
hdu.data_c = cutout.data
plt.contour(hdu.data_c, levels=[-0.003*0.327, 0.003*0.327, 0.006*0.327, 0.012*0.327, 0.024*0.327, 0.048*0.327, 0.096*0.327, 
                                0.192*0.327, 0.348*0.327, 0.768*0.327], colors='black')

# ----------------- Visualize map --------------
plt.tick_params(axis='both', labelsize = 25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Chi_square_map.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
