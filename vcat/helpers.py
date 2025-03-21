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
from numpy import linalg
import scipy.ndimage
import scipy.signal
from scipy.interpolate import RegularGridInterpolator,griddata

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

    elif noise_method=="box":
        #determine image rms from box at the bottom left corner with size of 1/10th of the image dimension
        noise = 1.8*np.std(image[0:round(len(image)/10),0:round(len(image[0])/10)]) #factor 1.8 from self-cal errors
        levs1 = sigma_contour_limit * noise * np.logspace(0, 100, 100, endpoint=False, base=2)
        levs = np.flip(-levs1)
        levs = np.concatenate((levs, levs1))

    else:
        raise Exception("Please define valid noise method ('Histogram Fit', 'box', 'DIFMAP', 'Image RMS')")

    return levs, levs1

#gets components from .fits file
def getComponentInfo(filename,scale=60*60*1000):
    """Imports component info from a modelfit .fits file.
    Args:
        filename: Path to a modelfit (or clean) .fits file
    Returns:
        Pandas Dataframe with the model data (Flux, Delta_x, Delta_y, Major Axis, Minor Axis, PA, Typ_obj)    
    """

    if is_fits_file(filename):
        #read in fits file
        data_df = pd.DataFrame()
        hdu_list = fits.open(filename)
        comp_data = hdu_list[1].data
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
    else:
        #will assume that the file is a .mod file
        flux = np.array([])
        radius = np.array([])
        theta = np.array([])
        maj= np.array([])
        ratio = np.array([])
        pa = np.array([])
        typ_obj = np.array([])

        with open(filename, "r") as file:
            for line in file:
                if not line.startswith("!"):
                    linepart=line.split()
                    flux = np.append(flux,float(linepart[0].replace("v","")))
                    radius = np.append(radius,float(linepart[1].replace("v","")))
                    theta = np.append(theta,float(linepart[2].replace("v","")))
                    #other parameters might not be there, try
                    try:
                        maj = np.append(maj,float(linepart[3].replace("v","")))
                        ratio = np.append(ratio,float(linepart[4].replace("v","")))
                        pa = np.append(pa,float(linepart[5].replace("v","")))
                        typ_obj = np.append(typ_obj,1) # in this case it is a gaussian model component
                    except:
                        maj = np.append(maj,0)
                        ratio = np.append(ratio,0)
                        pa = np.append(pa,0)
                        typ_obj = np.append(typ_obj,0) #in this case it is a clean component
        #import completed now calculate additional parameters:
        delta_x=radius*np.sin(theta/180*np.pi)/scale
        delta_y=radius*np.cos(theta/180*np.pi)/scale
        maj=maj/scale
        min=ratio*maj

        #create data_df
        data_df = pd.DataFrame({'ratio': ratio, 'Minor_axis': min, 'Major_axis': maj, 'theta': theta, 'Delta_y': delta_y,
                                'Delta_x': delta_x ,"Flux": flux, "PA": pa, "Typ_obj": typ_obj})

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
            if delta_x[ind] > 0 and delta_y[ind]==0:
                theta.append(90)
            elif delta_x[ind] < 0 and delta_y[ind]==0:
                theta.append(-90)
            elif delta_x[ind] == 0 and delta_y[ind] < 0:
                theta.append(180)
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
def Jy2JyPerBeam(jpp,b_maj,b_min,px_inc):
    """Converts Jy/px to Jy/beam
        Args:
            jpp: Jansky per pixel
            b_maj: Major Axis
            b_min: Minor Axis
            px_inc: pixel size
        Returns:
            Jansky per pixel value
        """

    return jpp*PXPERBEAM(b_maj,b_min,px_inc)

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
    env = os.environ.copy()

    # add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        env['PATH'] = env['PATH'] + ':{0}'.format(difmap_path)

    #remove potential difmap boot files (we don't need them)
    env["DIFMAP_LOGIN"]=""
    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False,env=env)
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
    env = os.environ.copy()

    # add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        env['PATH'] = env['PATH'] + ':{0}'.format(difmap_path)

    # remove potential difmap boot files (we don't need them)
    env["DIFMAP_LOGIN"] = ""
    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False, env=env)
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

#gets common ellipse from point selection (needed for smallest common beam calculation), adapted from https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
def getMinVolEllipse(P=None, tolerance=0.1):
        """ Find the minimum volume ellipsoid which holds all the points

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Here, P is a numpy array of 2 dimensional points like this:
        P = [[x,y], <-- one point per line
             [x,y],
             [x,y]]

        Returns:
        (center, radii, rotation)

        """

        (N, d) = np.shape(P)
        d = float(d)

        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T

        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse
        center = np.dot(P.T, u)

        # the A matrix for the ellipse
        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) -
                       np.array([[a * b for b in center] for a in center])
                       ) / d
        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)

        return (center, radii, rotation)

def get_common_beam(majs,mins,posas,arg='common',ppe=100,tolerance=0.0001,plot_beams=False):
    '''Derive the beam to be used for the maps to be aligned.
    Args:
        majs: List of Major Axis Values
        mins: List of Minor Axis Values
        posas: List of Position Angles (in degrees)
        arg: Type of algorithm to use ("mean", "max", "median", "circ", "common")
        ppe: Points per Ellipse for "common" algorithm
        tolerance: Tolerance parameter for "common" algorithm
        plot_beams: Boolean to choose if a diagnostic plot of all beams and the common beam should be displayed
    Returns:
        [maj, min, pos] List with new major and minor axis and position angle
    '''

    if arg=='mean':
        _maj = np.mean(majs)
        _min = np.mean(mins)
        _pos = np.mean(posas)
        sys.stdout.write(' Will use mean beam.\n')
    elif arg=='max':
        if np.argmax(majs)==np.argmax(mins):
            beam_ind=np.argmax(majs)
            _maj = majs[beam_ind]
            _min = mins[beam_ind]
            _pos = posas[beam_ind]
        else:
            print('could not derive max beam, defaulting to common beam.\n')
            return get_common_beam(majs,mins,posas,arg="common")
        sys.stdout.write(' Will use max beam.\n')
    elif arg=='median':
        _maj = np.median(majs)
        _min = np.median(mins)
        _pos = np.median(posas)
        sys.stdout.write(' Will use median beam.\n')
    elif arg == 'circ':
        _maj = np.median(majs)
        _min = _maj
        _pos = 0
    elif arg == 'common':
        if plot_beams:
            fig = plt.figure()
            ax = fig.add_subplot()

        sample_points = np.empty(shape=(ppe * len(majs), 2))
        for ind in range(len(majs)):
            bmaj = majs[ind]
            bmin = mins[ind]
            posa = posas[ind]/180*np.pi

            if len(majs) == 1:
                return bmaj, bmin, posa

            # sample ellipse points
            ellipse_angles = np.linspace(0, 2 * np.pi, ppe)
            X = -bmin / 2 * np.sin(ellipse_angles)
            Y = bmaj / 2 * np.cos(ellipse_angles)

            # rotate them according to position angle
            X_rot = -X * np.cos(posa) - Y * np.sin(posa)
            Y_rot = X * np.sin(posa) + Y * np.cos(posa)

            for i in range(ppe):
                sample_points[ind * ppe + i] = np.array([X_rot[i], Y_rot[i]])
            if plot_beams:
                plt.plot(X_rot, Y_rot, c="k")

        # find minimum ellipse
        (center, radii, rotation) = getMinVolEllipse(sample_points, tolerance=tolerance)

        # find out bmaj, bmin and posa
        bmaj_ind = np.argmax(radii)

        if bmaj_ind == 0:
            bmaj = 2 * radii[0]
            bmin = 2 * radii[1]
            posa = -np.arcsin(rotation[1][0]) / np.pi * 180 - 90
        else:
            bmaj = 2 * radii[1]
            bmin = 2 * radii[0]
            posa = -np.arcsin(rotation[1][0]) / np.pi * 180

        # make posa from -90 to +90
        if posa > 90:
            posa = posa - 180
        elif posa < -90:
            posa = posa + 180

        # plot ellipsoid
        if plot_beams:
            from matplotlib import patches
            ellipse = patches.Ellipse(center, bmin, bmaj, angle=posa, fill=False, zorder=2, linewidth=2, color="r")
            ax.add_patch(ellipse)

            ax.axis("equal")
            plt.show()

        _maj = bmaj
        _min = bmin
        _pos = posa
    else:
        raise Exception("Please use a valid arg value ('common', 'max', 'median', 'mean', 'circ')")


    common_beam=[_maj,_min,_pos]
    sys.stdout.write("{} beam calculated: {}\n".format(arg,common_beam))
    return common_beam

def elliptical_gaussian_kernel(size_x, size_y, sigma_x, sigma_y, theta):
    """Generate an elliptical Gaussian kernel with rotation."""
    y, x = np.meshgrid(np.linspace(-size_y//2, size_y//2, size_y), np.linspace(-size_x//2, size_x//2, size_x))

    # Rotation matrix
    theta = np.deg2rad(theta)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # Elliptical Gaussian formula
    g = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))
    return g / np.sum(g)  # Normalize the kernel

def convolve_with_elliptical_gaussian(image, sigma_x, sigma_y, theta):
    """Convolves a 2D image with an elliptical Gaussian kernel."""
    kernel = elliptical_gaussian_kernel(image.shape[1], image.shape[0], sigma_x, sigma_y, theta)
    convolved = scipy.signal.fftconvolve(image, kernel, mode='same')
    return convolved

def get_frequency(filepath):
    with fits.open(filepath) as hdu_list:
        return float(hdu_list[0].header["CRVAL3"])

def sort_fits_by_date_and_frequency(fits_files):
    if fits_files!="":
        fits_files = np.array(fits_files)

        if len(fits_files) > 0:
            dates = np.array([get_date(f) for f in fits_files])
            frequencies = np.array([get_frequency(f) for f in fits_files])

            # Sort primarily by date, secondarily by frequency
            sorted_indices = np.lexsort((frequencies, dates))
            fits_files = fits_files[sorted_indices]

        return fits_files.tolist()
    else:
        return fits_files

def get_uvf_frequency(filepath):
    """Extracts frequency from the FITS header by finding the correct CVALX."""
    with fits.open(filepath) as hdu_list:
        header = hdu_list[0].header
        for i in range(1, 100):  # Check CTYPE1 to CTYPE99 (assuming X is within this range)
            ctype_key = f"CTYPE{i}"
            cval_key = f"CRVAL{i}"
            if ctype_key in header and "FREQ" in header[ctype_key]:
                return float(header[cval_key])
        raise ValueError(f"Frequency keyword not found in {filepath}")

def sort_uvf_by_date_and_frequency(uvf_files):
    if uvf_files!="":
        uvf_files = np.array(uvf_files)

        if len(uvf_files) > 0:
            dates = np.array([fits.open(f)[0].header["DATE-OBS"].split("T")[0] for f in uvf_files])
            frequencies = np.array([get_uvf_frequency(f) for f in uvf_files])
            # Sort by date first, then by frequency
            sorted_indices = np.lexsort((frequencies, dates))
            uvf_files = uvf_files[sorted_indices]

        return uvf_files.tolist()
    else:
        return uvf_files

def closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

def func_turn(x, i0, turn, alpha0, alphat = 2.5):
    """Turnover frequency function."""
    return i0 * (x / turn)**alphat * (1.0 - np.exp(-(turn / x)**(alphat - alpha0)))

def plot_pixel_fit(frequencies, brightness, err_brightness, fitted_func, pixel, popt, peak_brightness):
    """Plot the data points and fitted function for a specific pixel."""
    x_smooth = np.linspace(min(frequencies), max(frequencies), 10000)  # High-resolution x-axis
    y_smooth = func_turn(x_smooth, *popt)  # Fitted function for high-res x-axis
    plt.figure(figsize=(10, 6))
    plt.style.use('default')
    plt.errorbar(frequencies, brightness, yerr=err_brightness, fmt='o', color='blue', label='Data Points')
    plt.plot(x_smooth, y_smooth, color='red', label=f'Fitted Function\nPeak: {peak_brightness:.2f} GHz')
    plt.xlabel('Frequency [GHz]', fontsize=16)
    plt.ylabel('Brightness [Jy/beam]', fontsize=16)
    plt.title(f'Pixel ({pixel[1]}, {pixel[0]})', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    #plt.savefig(f'pixel_fit_{pixel[1]}_{pixel[0]}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def rotate_points(x, y, angle_deg):
    """ Rotate points (x, y) by angle (in degrees) around the origin. """
    angle_rad = np.radians(angle_deg)  # Convert degrees to radians
    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

    # Apply rotation matrix
    x_new = cos_theta * x - sin_theta * y
    y_new = sin_theta * x + cos_theta * y

    return x_new, y_new

def rotate_mod_file(mod_file,angle, output="tmp.mod"):
    with open(mod_file) as infile, open("tmp.mod","w") as outfile:
        for line in infile:
            if not line.startswith("#"):
                linepart = line.split()
                posa=float(linepart[2])
                posa+=angle

                if posa<-180:
                    posa+=360
                elif posa>180:
                    posa-=360

                linepart[2]="{:.3f}".format(posa)
                outfile.write(" ".join(linepart)+"\n")

    os.replace("tmp.mod",output)

def rotate_uvf_file(uvf_file,angle, output):
    with(fits.open(uvf_file)) as f:
        u=f[0].data["UU"]
        v=f[0].data["VV"]
        new_u, new_v = rotate_points(u,v,angle)
        f[0].data["UU"]=new_u
        f[0].data["VV"]=new_v
        f[1].header["XTENSION"]="BINTABLE"
        f[2].header["XTENSION"]="BINTABLE"
        f.writeto(output,overwrite=True)

def is_fits_file(filename):
    try:
        with fits.open(filename) as hdul:
            return True  # Successfully opened, it's a FITS file
    except Exception:
        return False

def convert_image_to_polar(X,Y,Z,nrad="",ntheta=361):
    X, Y = np.meshgrid(X, Y)

    r_max = np.sqrt((X.max() - X.min()) ** 2 + (Y.max() - Y.min()) ** 2) / 2 /np.sqrt(2)-10
    if nrad=="":
        nrad=int(len(X)/2)
    r = np.linspace(0, r_max, nrad)
    theta = np.linspace(0, 2 * np.pi, ntheta)
    R, Theta = np.meshgrid(r, theta)

    # Convert Polar back to Cartesian
    X_polar = R * np.cos(Theta)
    Y_polar = R * np.sin(Theta)

    # Flatten for interpolation
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = Z.ravel()
    polar_points = np.column_stack((X_polar.ravel(), Y_polar.ravel()))

    # Interpolate
    Z_polar = griddata(points, values, polar_points, method='cubic')
    Z_polar = Z_polar.reshape(R.shape)

    Theta = -Theta / np.pi * 180 + 90
    Theta = np.where(Theta < -180, Theta + 360, Theta)

    #roll it to start with -180 and end with +180 in theta
    Theta=np.flip(Theta,axis=0)
    ind=np.argmin(Theta[:,0])
    Theta=np.roll(Theta,shift=-ind,axis=0)
    Z_polar=np.flip(Z_polar,axis=0)
    Z_polar=np.roll(Z_polar,shift=-ind,axis=0)

    return R, Theta, Z_polar