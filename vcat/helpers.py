from os import write
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from astropy.io import fits
from astropy.modeling import models, fitting
import os
from astropy.time import Time
import sys
import pexpect
from datetime import datetime
import colormaps as cmaps
import matplotlib.ticker as ticker

# takes a an image (2d) array as input and calculates the sigma levels for plotting, sigma_contour_limit denotes the sigma level of the lowest contour
def get_sigma_levs(image,  # 2d array/list
                   sigma_contour_limit=3, # choose the lowest sigma contour to plot
                   noise_method="Image RMS",
                   noise=0,
                   ):
    if noise_method=="Histogram Fit":
        Z1 = image.flatten()
        bin_heights, bin_borders = np.histogram(Z1 - np.min(Z1) + 10 ** (-5), bins="auto")
        bin_widths = np.diff(bin_borders)
        bin_centers = bin_borders[:-1] + bin_widths / 2.
        bin_heights_err = np.where(bin_heights != 0, np.sqrt(bin_heights), 1)

        t_init = models.Gaussian1D(np.max(bin_heights), np.median(Z1 - np.min(Z1) + 10 ** (-5)), 0.001)
        fit_t = fitting.LevMarLSQFitter()
        t = fit_t(t_init, bin_centers, bin_heights, weights=1. / bin_heights_err)
        noise = t.stddev.value

        # Set contourlevels to mean value + 3 * rms_noise * 2 ** x
        levs1 = t.mean.value + np.min(Z1) - 10 ** (-5) + sigma_contour_limit * noise * np.logspace(0, 100, 100,
                                                                                                            endpoint=False,
                                                                                                            base=2)
        levs = t.mean.value + np.min(Z1) - 10 ** (-5) - sigma_contour_limit * noise * np.logspace(0, 100, 100,
                                                                                                           endpoint=False,
                                                                                                           base=2)
        levs = np.flip(levs)
        levs = np.concatenate((levs, levs1))

    elif noise_method=="Image RMS":
        Z1 = image.flatten()
        noise = np.std(Z1)
        levs1 = sigma_contour_limit * noise * np.logspace(0, 100, 100, endpoint=False, base=2)
        levs = np.flip(-levs1)
        levs = np.concatenate((levs, levs1))

    elif noise_method=="DIFMAP":
        levs1 = sigma_contour_limit * noise * np.logspace(0, 100, 100, endpoint=False, base=2)
        levs = np.flip(-levs1)
        levs = np.concatenate((levs, levs1))



    return levs, levs1

#gets components from .fits file
def getComponentInfo(filename,scale=60*60*1000):
    """Imports component info from a modelfit .fits file.
    Args:
        filename: Path to a modelfit (or clean) .fits file
    Returns:
        Pandas Dataframe with the model data (Flux, Delta_x, Delta_y, Major Axis, Minor Axis, PA, Typ_obj)    
    """

    #TODO also include reading .mod files

    data_df = pd.DataFrame()
    hdu_list = fits.open(filename)
    comp_data = hdu_list[1].data
    freq=hdu_list[0].header["CRVAL3"]
    comp_data1 = np.zeros((len(comp_data), len(comp_data[0])))
    date = np.array([])
    year = np.array([])
    mjd = np.array([])
    for j in range(len(comp_data)):
        comp_data1[j, :] = comp_data[j]
        date1=get_date(filename)
        date = np.append(date, date1)
        t = Time(date1)
        year = np.append(year, t.jyear)
        mjd = np.append(mjd, t.mjd)
    comp_data1_df = pd.DataFrame(data=comp_data1,
                                 columns=["Flux", "Delta_x", "Delta_y", "Major_axis", "Minor_axis", "PA",
                                          "Typ_obj"])
    comp_data1_df["Date"] = date
    comp_data1_df["Year"] = year
    comp_data1_df["mjd"] = mjd
    comp_data1_df.sort_values(by=["Delta_x", "Delta_y"], ascending=False, inplace=True)
    if data_df.empty:
        data_df = comp_data1_df
    else:
        data_df = pd.concat([data_df, comp_data1_df], axis=0, ignore_index=True)
    os.makedirs("tmp",exist_ok=True)

    #write Radius, ratio and Angle also to database
    data_df['radius'] = np.sqrt(data_df['Delta_x'] ** 2 + data_df['Delta_y'] ** 2) * scale

    # Function to calculate 'theta'
    def calculate_theta(row):
        if (row['Delta_y'] > 0 and row['Delta_x'] > 0) or (row['Delta_y'] > 0 and row['Delta_x'] < 0):
            return np.arctan(row['Delta_x'] / row['Delta_y']) / np.pi * 180
        elif (row['Delta_y'] < 0 and row['Delta_x'] > 0):
            return np.arctan(row['Delta_x'] / row['Delta_y']) / np.pi * 180 + 180
        elif (row['Delta_y'] < 0 and row['Delta_x'] < 0):
            return np.arctan(row['Delta_x'] / row['Delta_y']) / np.pi * 180 - 180
        return 0

    # Apply function to calculate 'theta'
    data_df['theta'] = data_df.apply(calculate_theta, axis=1)

    # Calculate 'ratio'
    data_df['ratio'] = data_df.apply(lambda row: row['Minor_axis'] / row['Major_axis'] if row['Major_axis'] > 0 else 0,
                                 axis=1)

    return data_df

#writes a .mod file given an input of from getComponentInfo(fitsfile)
#the adv options adds a "v" character to the model to make the parameters fittable in DIFMAP
def write_mod_file(model_df,writepath,freq,scale=60*60*1000,adv=False):
    """writes a .mod file given an input DataFrame with component info.
    Args:
        model_df: DataFrame with model component info (e.g. generated by getComponentInfo())
        writepath: Filepath where to write the .mod file
        freq: Frequency of the observation in GHz
        scale: Conversion of the image scale to degrees (default milli-arc-seconds -> 60*60*1000)
    Returns:
        Nothing, but writes a .mod file to writepath
    """

    flux = np.array(model_df["Flux"])
    delta_x = np.array(model_df["Delta_x"])
    delta_y = np.array(model_df["Delta_y"])
    maj = np.array(model_df["Major_axis"])
    min = np.array(model_df["Minor_axis"])
    pos = np.array(model_df["PA"])
    typ_obj = np.array(model_df["Typ_obj"])

    original_stdout=sys.stdout
    sys.stdout=open(writepath,'w')

    #TODO this part is probably redundant since it is now already included in the model_df
    radius=[]
    theta=[]
    ratio=[]

    for ind in range(len(flux)):
        radius.append(np.sqrt(delta_x[ind]**2+delta_y[ind]**2)*scale)
        if (delta_y[ind]>0 and delta_x[ind]>0) or (delta_y[ind]>0 and delta_x[ind]<0):
            theta.append(np.arctan(delta_x[ind]/delta_y[ind])/np.pi*180)
        elif delta_y[ind]<0 and delta_x[ind]>0:
            theta.append(np.arctan(delta_x[ind]/delta_y[ind])/np.pi*180+180)
        elif delta_y[ind]<0 and delta_x[ind]<0:
            theta.append(np.arctan(delta_x[ind] / delta_y[ind]) / np.pi * 180 - 180)
        else:
            theta.append(0)
        if maj[ind]>0:
            ratio.append(min[ind]/maj[ind])
        else:
            ratio.append(0)

    #sort by flux
    argsort=flux.argsort()[::-1]
    flux=np.array(flux)[argsort]
    radius=np.array(radius)[argsort]
    theta=np.array(theta)[argsort]
    maj=np.array(maj)[argsort]
    ratio=np.array(ratio)[argsort]
    pos=np.array(pos)[argsort]
    typ_obj=np.array(typ_obj)[argsort]

    #check if we need to ad "v" to the components to make them fittable
    if adv:
        ad="v"
    else:
        ad=""

    for ind in range(len(flux)):
        print(" "+"{:.8f}".format(flux[ind])+ad+"   "+
              "{:.8f}".format(radius[ind])+ad+"    "+
              "{:.3f}".format(theta[ind])+ad+"   "+
              "{:.7f}".format(maj[ind]*scale)+ad+"    "+
              "{:.6f}".format(ratio[ind])+"   "+
              "{:.4f}".format(pos[ind])+"  "+
              str(int(typ_obj[ind]))+" "+
              "{:.5E}".format(freq)+"   0")

    sys.stdout = original_stdout

def write_mod_file_from_casa(file_path,channel="i",export="export.mod"):
    """Writes a .mod file from a CASA exported .fits model file.
    Args:
        file_path: File path to a .fits model file as exported from a CASA .model file (e.g. with exportfits() in CASA)
        channel: Choose the Stokes channel to use (options: "i","q","u","v")
        export: File path where to write the .mod file
    Returns:
        Nothing, but writes a .mod file to export
    """

    image_data=ImageData(file_path)
    if channel=="i":
        clean_map=image_data.Z
    elif channel=="q":
        clean_map=image_data.stokes_q
    elif channel=="u":
        clean_map=image_data.stokes_u
    else:
        raise Exception("Please enter a valid channel (i,q,u)")

    #read out clean components from pixel map
    delta_x=[]
    delta_y=[]
    flux=[]
    zeros=[]
    for i in range(len(image_data.X)):
        for j in range(len(image_data.Y)):
            if clean_map[j][i]>0:
                delta_x.append(image_data.X[i]/image_data.scale)
                delta_y.append(image_data.Y[j]/image_data.scale)
                flux.append(clean_map[j][i])
                zeros.append(0.0)

    #create model_df
    model_df=pd.DataFrame(
        {'Flux': flux,
         'Delta_x': delta_x,
         'Delta_y': delta_y,
         'Major_axis': zeros,
         'Minor_axis': zeros,
         'PA': zeros,
         'Typ_obj': zeros
         })

    #create mod file
    write_mod_file(model_df,export,image_data.freq,image_data.scale)

def get_freq(fits_file):
    freq=0
    hdu_list=fits.open(fits_file)
    for i in range(1,4):
        try:
            if "FREQ" in hdu_list[0].header["CTYPE"+str(i)]:
                freq=hdu_list[0].header["CRVAL"+str(i)]
        except:
            pass
    return float(freq)

def get_date(filename):
    """Returns the date of an observation from a .fits file.
    Args:
        filename: Path to the .fits file
    Returns:
        Date in the format year-month-day
    """

    hdu_list=fits.open(filename)
    # Plot date
    time = hdu_list[0].header["DATE-OBS"]
    time = time.split("T")[0]
    time = time.split("/")
    if len(time) == 1:
        date = time[0]
    elif len(time) == 3:
        if len(time[0]) < 2:
            day = "0" + time[0]
        else:
            day = time[0]
        if len(time[1]) < 2:
            month = "0" + time[1]
        else:
            month = time[1]
        if len(time[2]) == 2:
            if 45 < int(time[2]) < 100:
                year = "19" + time[2]
            elif int(time[2]) < 46:
                year = "20" + time[2]
        elif len(time[2]) == 4:
            year = time[2]
        date = year + "-" + month + "-" + day
    return date

#needs a mod_file as input an returns the total flux
def total_flux_from_mod(mod_file,squared=False):
    """needs a mod_file as input an returns the total flux
    Args:
        mod_file: Path to a .mod file
        squared: If true, returns the sum of the squared fluxes (useful for polarization)
    Returns:
        The total flux in the .mod file (usually in mJy, depending on the .mod file)
    """

    lines=open(mod_file).readlines()
    total_flux=0
    for line in lines:
        if not line.startswith("!"):
            linepart=line.split()
            if squared:
                total_flux+=float(linepart[0])**2
            else:
                total_flux+=float(linepart[0])
    return total_flux
                
def PXPERBEAM(b_maj,b_min,px_inc):
    """calculates the pixels per beam.
    Args:
        b_maj: major axis
        b_min: minor axis
        px_inc: pixel size
    Returns:
        pixels per beam
    """

    beam_area = np.pi/(4*np.log(2))*b_min*b_maj
    PXPERBEAM = beam_area/(px_inc**2)
    return PXPERBEAM
#

def JyPerBeam2Jy(jpb,b_maj,b_min,px_inc):
    """Converts Jy/beam to Jy
    Args:
        jbp: Jansky per beam value
        b_maj: Major Axis
        b_min: Minor Axis
        px_inc: pixel size
    Returns:
        Jansky value
    """

    return jpb/PXPERBEAM(b_maj,b_min,px_inc)

# calculates the image noise from the residual map in a given box area
def get_residual_map(uvf_file,mod_file, difmap_path, save_location="residual.fits", npix=2048,pxsize=0.05):
    """ calculates residual map and stores it as .fits file.
    Args:
        uvf_file: Path to a .uvf file
        mod_file: Path to a .mod file
        difmap_path: Path to the DIFMAP executable
        save_location: Path where to store the residual map .fits file
        npix: Number of pixels to use
        pxsize: Pixel Size (usually in mas)
    Returns:
        Nothing, but writes a .fits file including the residual map
    """

    # add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ':{0}'.format(difmap_path)

    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False)
    child.expect_exact("0>", None, 2)

    def send_difmap_command(command, prompt="0>"):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)

    send_difmap_command("obs "+uvf_file)
    send_difmap_command("select i")
    send_difmap_command("uvw 0,-1")  # use natural weighting
    send_difmap_command("rmod "+mod_file)
    send_difmap_command("maps " + str(npix) + "," + str(pxsize))
    send_difmap_command("wdmap " + save_location) #save the residual map to a fits file

    os.system("rm -rf difmap.log*")

def get_noise_from_residual_map(residual_fits, center_x, center_y, x_width, y_width,scale=0):
    """calculates the noise from the residual map in a given box
    Args:
        residual_fits: Path to .fits file with residual map
        center_x: X-center of the box to use for noise calculation in pixels
        center_y: Y-center of the box to use for noise calculation in pixels
        x_width: X-width of the box in pixels
        y_width: Y-width of the box in pixels
    Returns:
        Noise in the given box from the residual map
    """

    residual_map = ImageData(residual_fits)
    data=residual_map.Z

    x_max=np.argmin(abs(residual_map.X*scale-(center_x-x_width/2)))
    y_min=np.argmin(abs(residual_map.Y*scale-(center_y-y_width/2)))
    x_min=np.argmin(abs(residual_map.X*scale-(center_x+x_width/2)))
    y_max=np.argmin(abs(residual_map.Y*scale-(center_y+y_width/2)))

    return np.std(data[x_min:x_max,y_min:y_max]) #TODO check order of x/y here and if AVERAGE is the correct thing to do!!!

#returns the reduced chi-square of a modelfit
def get_model_chi_square_red(uvf_file,mod_file,difmap_path):
    # add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + ':{0}'.format(difmap_path)

    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False)
    child.expect_exact("0>", None, 2)

    def send_difmap_command(command, prompt="0>"):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)
        return child.before

    send_difmap_command("obs " + uvf_file)
    send_difmap_command("select i")
    send_difmap_command("uvw 0,-1")  # use natural weighting
    send_difmap_command("rmod " + mod_file)
    #send modelfit 0 command to calculate chi-squared
    output=send_difmap_command("modelfit 0")

    lines=output.splitlines()
    for line in lines:
        if "Iteration 00" in line:
            chi_sq_red=float(line.split("=")[1].split()[0])

    os.system("rm -rf difmap.log")
    return chi_sq_red


def format_scientific(number):
    # Format number in scientific notation
    sci_str = "{:.0e}".format(number)

    # Split into mantissa and exponent
    mantissa, exp = sci_str.split('e')

    # Convert exponent to integer
    exp = int(exp)

    # Unicode superscript mapping
    superscript = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }

    # Handle negative exponents
    if exp < 0:
        exp_str = '⁻' + ''.join(superscript.get(digit, digit) for digit in str(abs(exp)))
    else:
        exp_str = ''.join(superscript.get(digit, digit) for digit in str(exp))

    # Format the result
    result = f"{mantissa} × 10{exp_str}"

    return result
