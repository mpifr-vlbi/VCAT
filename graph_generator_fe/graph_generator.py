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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time
import sys
import pexpect

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = 'Quicksand'
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=dp(13)
font_size_axis_tick=dp(12)

class KinematicPlot(object):
    def __init__(self):

        super().__init__()
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)

    def plot_kinematics(self,component_collection,color):
        if component_collection.length()>0:
            self.ax.scatter(component_collection.year,component_collection.dist,c=color,marker=".")
        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Distance from Core [mas]', fontsize=font_size_axis_title)

    def plot_fluxs(self,component_collection,color):
        if component_collection.length() > 0:
            self.ax.plot(component_collection.year, component_collection.fluxs, c=color, label=component_collection.name,marker=".")
        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Flux Density [Jy]', fontsize=font_size_axis_title)

    def plot_tbs(self,component_collection,color):
        if component_collection.length() > 0:
            lower_limit_inds = np.where(np.array(component_collection.tbs_lower_limit))[0]
            tb_value_inds = np.where(np.array(component_collection.tbs_lower_limit)==False)[0]
            self.ax.plot(np.array(component_collection.year)[tb_value_inds],
                    np.array(component_collection.tbs)[tb_value_inds], c=color, label=component_collection.name,marker=".")
            self.ax.scatter(np.array(component_collection.year)[lower_limit_inds],
                    np.array(component_collection.tbs)[lower_limit_inds], c=color, marker="^")

            self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Brightness Temperature [K]', fontsize=font_size_axis_title)
        self.ax.set_yscale("log")

    def set_limits(self,x,y):
        self.ax.set_xlim(x)
        self.ax.set_ylim(y)
    def plot_linear_fit(self,x_min,x_max,slope,y0,color,label=""):
        def y(x):
            return slope*x+y0
        self.ax.plot([x_min,x_max],[y(x_min),y(x_max)],color,label=label)

class ImageData(object):
    """ Class to handle VLBI Image data (single image with or without polarization)

    Attributes:
        fits_file: Path to a .fits file containing the data (Stokes-I (DIFMAP) or Full Polarization (CASA)).
        uvf_file: Path to a .uvf file corresponding to the fits_file
        freq: Frequency of the Image in GHz
        stokes_i: Optional input of a 2d-array with Stokes-I values
        model: Path to a .fits file including the 
        lin_pol: Optional input of a 2d-array with lin-pol values
        evpa: Optional input of a 2d-array with EVPA values
        pol_from_stokes: Select whether to read in polarization from stokes_q/u or lin_pol/evpa
        stokes_q: Path to a .fits file containing Stokes-Q data
        stokes_u: Path to a .fits file containing Stokes-U data
        model_save_dir: Path where to store created .mod files 
        is_casa_model: If a model .fits from CASA was imported, set to True, otherwise set to False
        difmap_path: Provide the path to your difmap executable
    """


    def __init__(self,
            fits_file="",
            uvf_file="",
            stokes_i=[],
            model="",
            lin_pol=[],
            evpa=[],
            pol_from_stokes=True,
            stokes_q="",
            stokes_u="",
            model_save_dir="tmp/mod_files_model/",
            is_casa_model=False,
            difmap_path=""):

        self.file_path = fits_file
        self.model_file_path = model
        self.lin_pol=lin_pol
        self.evpa=evpa
        self.stokes_i=stokes_i
        self.uvf_file=uvf_file
        self.difmap_path=difmap_path
        self.residual_map_path=""

        # Read clean files in
        if fits_file!="":
            hdu_list=fits.open(fits_file)
            self.hdu_list = hdu_list
            self.no_fits=False
        else:
            self.no_fits=True

        stokes_q_path=stokes_q
        stokes_u_path=stokes_u
        #read stokes data from input files if defined
        if stokes_q != "":
            try:
                stokes_q = fits.open(stokes_q)[0].data[0, 0, :, :]
            except:
                stokes_q=stokes_q
        else:
            stokes_q=[]

        if stokes_u != "":
            try:
                stokes_u = fits.open(stokes_u)[0].data[0, 0, :, :]
            except:
                stokes_u = stokes_u
        else:
            stokes_u=[]

        self.stokes_u=stokes_u
        self.stokes_q=stokes_q

        # Set name
        self.name = hdu_list[0].header["OBJECT"]

        self.freq = float(hdu_list[0].header["CRVAL3"])  # frequency in Hertz

        # Unit selection and adjustment
        self.degpp = abs(hdu_list[0].header["CDELT1"])  # degree per pixel

        if self.degpp > 0.01:
            self.unit = 'deg'
            self.scale = 1.
        elif self.degpp > 6.94e-6:
            self.unit = 'arcmin'
            self.scale = 60.
        elif self.degpp > 1.157e-7:
            self.scale = 60. * 60.
            self.unit = 'arcsec'
        else:
            self.scale = 60. * 60. * 1000.
            self.unit = 'mas'

        # Convert Pixel into unit
        self.X = np.linspace(0, hdu_list[0].header["NAXIS1"], hdu_list[0].header["NAXIS1"],
                endpoint=False)  # NAXIS1: number of pixels at R.A.-axis
        for j in range(len(self.X)):
            self.X[j] = (self.X[j] - hdu_list[0].header["CRPIX1"]) * hdu_list[0].header[
                    "CDELT1"] * self.scale  # CRPIX1: reference pixel, CDELT1: deg/pixel
            self.X[int(hdu_list[0].header["CRPIX1"])] = 0.0

        self.Y = np.linspace(0, hdu_list[0].header["NAXIS2"], hdu_list[0].header["NAXIS2"],
                endpoint=False)  # NAXIS2: number of pixels at Dec.-axis
        for j in range(len(self.Y)):
            self.Y[j] = (self.Y[j] - hdu_list[0].header["CRPIX2"]) * hdu_list[0].header[
                    "CDELT2"] * self.scale  # CRPIX2: reference pixel, CDELT2: deg/pixel
            self.Y[int(hdu_list[0].header["CRPIX2"])] = 0.0

        self.extent = np.max(self.X), np.min(self.X), np.min(self.Y), np.max(self.Y)

        if not self.no_fits:
            self.image_data = hdu_list[0].data
            self.Z = self.image_data[0, 0, :, :]

        else:
            try:
                self.Z=self.stokes_i
            except:
                pass

        #overwrite fits image data with stokes_i input if given
        if not stokes_i==[]:
            self.Z=stokes_i

        #read in polarization input

        # check if FITS file contains more than just Stokes I
        only_stokes_i = False
        if hdu_list[0].data.shape[0] == 1:
            only_stokes_i = True
        if (np.shape(self.Z) == np.shape(stokes_q) and np.shape(self.Z) == np.shape(stokes_u) and
                np.shape(stokes_q) == np.shape(stokes_u)):
            only_stokes_i = True #in this case override the polarization data with the data that was input to Q and U

        if only_stokes_i:
            #DIFMAP Style
            pols=1
            # Check if linpol/evpa/stokes_i have same dimensions!
            dim_wrong = True
            if pol_from_stokes:
                if (np.shape(self.Z) == np.shape(stokes_q) and np.shape(self.Z) == np.shape(stokes_u) and
                        np.shape(stokes_q) == np.shape(stokes_u)):
                    dim_wrong = False
                    self.stokes_q=stokes_q
                    self.stokes_u=stokes_u
                else:
                    self.lin_pol = np.zeros(np.shape(self.Z))
                    self.evpa = np.zeros(np.shape(self.Z))
            else:
                if (np.shape(self.Z) == np.shape(lin_pol) and np.shape(self.Z) == np.shape(evpa) and
                        np.shape(lin_pol) == np.shape(evpa)):
                    dim_wrong = False
                    self.lin_pol=lin_pol
                    self.evpa=evpa
                else:
                    self.lin_pol=np.zeros(np.shape(self.Z))
                    self.evpa=np.zeros(np.shape(self.Z))
            self.image_data[0, 0, :, :] = self.Z
        else:
            #CASA STYLE
            pols=3
            dim_wrong=False
            self.stokes_q=hdu_list[0].data[1,0,:,:]
            self.stokes_u=hdu_list[0].data[2,0,:,:]
            self.image_data[1, 0, :, :] = self.stokes_q
            self.image_data[2, 0, :, :] = self.stokes_u

        if pol_from_stokes and not dim_wrong:
            self.lin_pol = np.sqrt(self.stokes_q ** 2 + self.stokes_u ** 2)
            self.evpa = 0.5 * np.arctan2(self.stokes_u, self.stokes_q)
            #shift to 0-180 (only positive)
            self.evpa[np.where(self.evpa<0)] = self.evpa[np.where(self.evpa<0)]+np.pi


        # Set beam parameters
        try:
            #DIFMAP style
            self.beam_maj = hdu_list[0].header["BMAJ"] * self.scale
            self.beam_min = hdu_list[0].header["BMIN"] * self.scale
            self.beam_pa = hdu_list[0].header["BPA"]
        except:
            try:
                #TODO check if this is actually working!
                #CASA style
                self.beam_maj, self.beam_min, self.beam_pa, na, nb = hdu_list[1].data[0]
                self.beam_maj=self.beam_maj*1000 #convert to mas
                self.beam_min=self.beam_min*1000 #convert to mas
            except:
                print("No input beam information!")
                self.beam_maj = 0
                self.beam_min = 0
                self.beam_pa = 0

        self.date = get_date(fits_file)


        #calculate image noise
        unused, levs_i = get_sigma_levs(self.Z, 1) #get noise for stokes i
        try:
            unused, levs_pol = get_sigma_levs(self.lin_pol, 1) #get noise for polarization
        except:
            levs_pol=[0]

        # calculate image noise
        unused, levs_i_3sigma = get_sigma_levs(self.Z, 3)  # get noise for stokes i
        try:
            unused, levs_pol_3sigma = get_sigma_levs(self.lin_pol, 3)  # get noise for polarization
        except:
            levs_pol_3sigma = [0]

        self.noise = levs_i[0]
        self.pol_noise = levs_pol[0]

        self.noise_3sigma  = levs_i_3sigma[0]
        self.pol_noise_3sigma = levs_pol_3sigma[0]

        #calculate integrated total flux in image
        self.integrated_flux_image = JyPerBeam2Jy(np.sum(self.Z), self.beam_maj, self.beam_min, self.degpp * self.scale)

        #calculate integrated pol flux in image
        self.integrated_pol_flux_image = JyPerBeam2Jy(np.sum(self.lin_pol),self.beam_maj,self.beam_min,self.degpp*self.scale)

        #calculate average EVPA (mask where lin pol < 3 sigma or stokes i < 3 sigma (same as in plot)
        integrate_evpa = np.ma.masked_where((self.lin_pol < self.pol_noise_3sigma) | (self.Z < self.noise_3sigma),
                self.evpa)
        self.evpa_average = np.average(integrate_evpa)

        if model!="" and not is_casa_model:
            #TODO basic checks if file is valid
            self.model=getComponentInfo(model)
            #write .mod file from .fits input
            os.makedirs(model_save_dir,exist_ok=True)
            write_mod_file(self.model, model_save_dir + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod", freq=self.freq)
        if is_casa_model:
            #TODO basic checks if file is valid
            os.makedirs(model_save_dir,exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_clean", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_q", exist_ok=True)
            os.makedirs(model_save_dir + "mod_files_u", exist_ok=True)
            write_mod_file_from_casa(self.file_path,channel="i", export=model_save_dir+"mod_files_clean/"+self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            write_mod_file_from_casa(self.file_path,channel="q", export=model_save_dir+"mod_files_q/"+self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            write_mod_file_from_casa(self.file_path,channel="u", export=model_save_dir+"mod_files_u/"+self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")

        else:
            self.model=None
        try:
            os.makedirs("tmp/mod_files_clean", exist_ok=True)
            os.makedirs("tmp/mod_files_q", exist_ok=True)
            os.makedirs("tmp/mod_files_u", exist_ok=True)
            #try to import model which is attached to the main .fits file
            model_i = getComponentInfo(fits_file)
            if self.model==None:
                self.model = model_i
            write_mod_file(model_i, "tmp/mod_files_clean/"+ self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod", freq=self.freq)
            #load stokes q and u clean models
            model_q=getComponentInfo(stokes_q_path)
            write_mod_file(model_q, "tmp/mod_files_q/" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod", freq=self.freq)
            model_u=getComponentInfo(stokes_u_path)
            write_mod_file(model_u, "tmp/mod_files_u/" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod", freq=self.freq)
        except:
            pass

        #calculate residual map if uvf and modelfile present
        if self.uvf_file!="" and self.model_file_path!="" and not is_casa_model:
            self.residual_map_path = model_save_dir + self.date + "_" + "{:.0f}".format(self.freq / 1e9).replace(".",
                    "_") + "GHz_residual.fits"
            get_residual_map(self.uvf_file,model_save_dir+ self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod",
                    difmap_path=self.difmap_path,
                    save_location=self.residual_map_path,npix=len(self.X),pxsize=self.degpp)

            hdu_list.close()

        #calculate cleaned flux density from mod files
        #first stokes I
        try:
            self.integrated_flux_clean=total_flux_from_mod("tmp/mod_files_clean/" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
        except:
            self.integrated_flux_clean = 0
        #and then polarization
        try:
            flux_q=total_flux_from_mod("tmp/mod_files_q/" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            flux_u=total_flux_from_mod("tmp/mod_files_u/" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            self.integrated_pol_flux_clean=np.sqrt(flux_u**2+flux_q**2)
        except:
            self.integrated_pol_flux_clean=0



class FitsImage(object):
    """Class that generates Matplotlib graph for a VLBI image.
    
    Attributes:
        image_data: ImageData object which includes the VLBI image.
        stokes_i_sigma_cut: Select the sigma cut to apply to Stokes I
        plot_mode: Choose which parameter to plot (options: "stokes_i","lin_pol","frac_pol")
        im_colormap: Choose whether to do colormap or not
        contour: Choose whether to do contour plot or not
        contour_color: Choose contour color
        contour_cmap: Choose colormap for contours
        contour_alpha: Choose transparency for contours
        contour_width: Choose width of contours
        im_color: Choose colormap name
        plot_beam: Choose whether to plot the beam or not
        overplot_gauss: Choose whether to overplot modelfit components (if available in image_data)
        component_color: Choose color to plot components
        overplot_clean: Choose whether to overplot clean components (if available in image_data)
        xlim: Choose X-plot limits
        ylim: Choose Y-plot limits
        plot_evpa: Choose whether to plot EVPAs or not
        evpa_width: Choose EVPA width
        evpa_len: Choose EVPA len in pixels
        lin_pol_sigma_cut: Choose lowest sigma contour for lin pol
        evpa_distance: Choose the distance of EVPA vectors to plot in pixels
        rotate_evpa: rotate EVPAs by a given angle in degrees (North through East)
        evpa_color: Choose EVPA color
        title: Choose plot title
        rcParams: Put in matplotlib rcParams for more modification to the plots        
    """

    def __init__(self,
            image_data, #ImageData object
            stokes_i_sigma_cut=3, #sigma_cut for stokes_i_contours
            plot_mode="stokes_i", #possible modes "stokes_i", "lin_pol", "frac_pol"
            im_colormap=False, #Choose whether to do colormap or not
            contour=True, #Choose whether to do contour plot or not
            contour_color = 'grey',  # input: array of color-strings; if None, the contour-colormap (contour_cmap) will be used
            contour_cmap = None,  # matplotlib colormap string
            contour_alpha = 1,  # transparency
            contour_width = 0.5,  # contour linewidth
            im_color='inferno', # string for matplotlib colormap
            plot_beam=True, #choose whether to plot beam or not
            overplot_gauss=False, #choose whether to plot modelfit components
            component_color="black", # choose component color for Gauss component
            overplot_clean=False, #choose whether to plot clean components
            xlim=[], #xplot limits, e.g. [5,-5]
            ylim=[], #yplot limits
            ###HERE STARTS POLARIZATION INPUT
            plot_evpa=False, #decide whether to plot EVPA or not
            evpa_width=2, #choose width of EVPA lines
            evpa_len=8,  # choose length of EVPA in pixels
            lin_pol_sigma_cut=3,  # choose lowest sigma contour for Lin Pol plot
            evpa_distance=10,  # choose distance of EVPA vectors to draw in pixels
            rotate_evpa=0, # rotate EVPAs by a given angle in degrees (North through East)
            evpa_color="white", # set EVPA color for plot
            title="", # plot title (default is date)
            rcparams={} # option to modify matplotlib look
            ):

        super().__init__()

        #read image
        self.clean_image = image_data
        self.clean_image_file = self.clean_image.file_path
        self.model_image_file = self.clean_image.model_file_path

        #set parameters
        self.name = self.clean_image.name
        self.freq = self.clean_image.freq
        image_data = self.clean_image.image_data
        X = self.clean_image.X
        Y = self.clean_image.Y
        Z = self.clean_image.Z
        unit = self.clean_image.unit
        scale = self.clean_image.scale
        degpp = self.clean_image.degpp
        extent = self.clean_image.extent
        date=self.clean_image.date
        self.evpa_width=evpa_width
        # Set beam parameters
        beam_maj = self.clean_image.beam_maj
        beam_min = self.clean_image.beam_min
        beam_pa = self.clean_image.beam_pa
        self.evpa_color=evpa_color

        #plot limits
        ra_max,ra_min,dec_min,dec_max=extent

        if len(xlim) == 2:
            ra_max, ra_min = xlim
        if len(ylim) == 2:
            dec_min, dec_max = ylim

        self.fig, self.ax = plt.subplots(1, 1)

        self.components=[]

        #component default color
        self.component_color = component_color

        fit_noise = True  # if True, the noise value and rms deviation will be fitted as described in the PhD-thesis of Moritz Böck (https://www.physik.uni-wuerzburg.de/fileadmin/11030400/Dissertation_Boeck.pdf); if False, the noise frome difmap will be used

        # Image colormap
        self.im_colormap = im_colormap  # if True, a image colormap will be done

        clean_alpha = 1  # float for sympol transparency

        #get sigma levs
        levs, levs1 = get_sigma_levs(Z,stokes_i_sigma_cut)

        # Image colormap
        if self.im_colormap == True and plot_mode=="stokes_i":
            self.plotColormap(Z,im_color,levs,levs1,extent)
            contour_color="white"


        if (plot_mode=="lin_pol" or plot_mode=="frac_pol") and np.sum(self.clean_image.lin_pol)!=0:

            levs_linpol, levs1_linpol = get_sigma_levs(self.clean_image.lin_pol, lin_pol_sigma_cut)

            if plot_mode=="lin_pol":
                self.plotColormap(self.clean_image.lin_pol,im_color,levs_linpol,levs1_linpol,extent,
                        label="Linear Polarized Intensity [Jy/beam]")
                if plot_mode=="frac_pol":
                    plot_lin_pol = np.array(self.clean_image.lin_pol)
                plot_frac_pol = plot_lin_pol / np.array(self.clean_image.Z)
                plot_frac_pol = np.ma.masked_where((plot_lin_pol < levs1_linpol[0]) | (self.clean_image.Z<levs1[0]),
                        plot_frac_pol)

                self.plotColormap(plot_frac_pol,im_color,np.zeros(100),[0.01],extent,
                        label="Fractional Linear Polarization")
                self.evpa_color="black"
                contour_color="grey"

        if plot_evpa and np.sum(self.clean_image.lin_pol)!=0:
            levs_linpol, levs1_linpol = get_sigma_levs(self.clean_image.lin_pol, lin_pol_sigma_cut)
            self.plotEvpa(self.clean_image.evpa, rotate_evpa, evpa_len, evpa_distance, levs1_linpol, levs1)

        # Contour plot
        if contour == True:
            if contour_cmap=="" or contour_cmap==None:
                contour_cmap=None
            else:
                contour_color=None

            self.ax.contour(X, Y, Z, linewidths=contour_width, levels=levs, colors=contour_color,
                    alpha=contour_alpha,
                    cmap=contour_cmap)

            # Set beam ellipse, sourcename and observation date positions
        size_x = np.absolute(ra_max) + np.absolute(ra_min)
        size_y = np.absolute(dec_max) + np.absolute(dec_min)
        if size_x > size_y:
            ell_x = ra_max - beam_maj
            ell_y = dec_min + beam_maj
        else:
            ell_x = ra_max - beam_maj
            ell_y = dec_min + beam_maj

        if plot_beam:
            # Plot beam
            beam = Ellipse([ell_x, ell_y], beam_maj, beam_min, -beam_pa + 90, fc='grey')
            self.ax.add_artist(beam)

        if title=="":
            self.ax.set_title(date + " " + "{:.0f}".format(self.freq/1e9)+" GHz", fontsize=font_size_axis_title)
        else:
            self.ax.set_title(title, fontsize=font_size_axis_title)

        # Read modelfit files in
        if (overplot_gauss == True) or (overplot_clean == True):
            model_df = getComponentInfo(self.model_image_file)

            # sort in gauss and clean components
            model_gauss_df = model_df #model_df[model_df["Major_axis"] > 0.].reset_index()
            model_clean_df = model_df[model_df["Major_axis"] == 0.].reset_index()

            # Overplot clean components
            if overplot_clean == True:
                c_x = model_clean_df["Delta_x"]
                c_y = model_clean_df["Delta_y"]
                c_flux = model_clean_df["Flux"]

                for j in range(len(c_x)):
                    if c_flux[j] < 0.:
                        self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='red', alpha=clean_alpha,
                                linewidth=0.2, zorder=2)
                    else:
                        self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='green', alpha=clean_alpha,
                                linewidth=0.2, zorder=2)

            # Overplot Gaussian components
            if overplot_gauss == True:

                g_x = model_gauss_df["Delta_x"]
                g_y = model_gauss_df["Delta_y"]
                g_maj = model_gauss_df["Major_axis"]
                g_min = model_gauss_df["Minor_axis"]
                g_pos = model_gauss_df["PA"]
                g_flux = model_gauss_df["Flux"]
                g_date = model_gauss_df["Date"]
                g_mjd = model_gauss_df["mjd"]
                g_year = model_gauss_df["Year"]

                for j in range(len(g_x)):
                    # plot component
                    self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale)
                    
        self.xmin,self.xmax = ra_min, ra_max
        self.ymin,self.ymax = dec_min, dec_max
        
        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)

        # Plot look tuning
        self.ax.set_aspect('equal', adjustable='box', anchor='C')
        self.ax.set_xlim(ra_min, ra_max)
        self.ax.set_ylim(dec_min, dec_max)
        self.ax.invert_xaxis()
        self.ax.set_xlabel('Relative R.A. [' + unit + ']',fontsize=font_size_axis_title)
        self.ax.set_ylabel('Relative DEC. [' + unit + ']',fontsize=font_size_axis_title)
        mpl.rcParams.update(rcparams)
        self.fig.tight_layout()

    def plotColormap(self,
                     Z, #2d data array to plot
                     im_color, #colormap to use
                     levs, #sigma levs output
                     levs1, #sigma levs output
                     extent, #plot lims x_min,x_max,y_min,y_max
                     label="Flux Density [Jy]" #label for colorbar
                     ):
        col = self.ax.imshow(Z, cmap=im_color, norm=colors.SymLogNorm(linthresh=levs1[0], linscale=0.5, vmin=levs1[0],
                                                                      vmax=0.5 * np.max(Z), base=10.), extent=extent,
                             origin='lower')
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = self.fig.colorbar(col, use_gridspec=True, cax=cax)
        cbar.set_label(label)

    def plotComponent(self,x,y,maj,min,pos,scale):

        # Plotting ellipses
        comp = Ellipse([x * scale, y * scale], maj * scale, min * scale, -pos + 90,
                       fill=False, zorder=2, color=self.component_color, lw=0.5)
        ellipse=self.ax.add_artist(comp)

        #deal with point like components
        if maj==0 and min==0:
            maj=0.1/scale
            min=0.1/scale

        # Plotting axes of the ellipses
        maj1_x = x - np.sin(-np.pi / 180 * pos) * maj * 0.5
        maj1_y = y + np.cos(-np.pi / 180 * pos) * maj * 0.5
        maj2_x = x + np.sin(-np.pi / 180 * pos) * maj * 0.5
        maj2_y = y - np.cos(-np.pi / 180 * pos) * maj * 0.5

        min1_x = x - np.sin(-np.pi / 180 * (pos + 90)) * min * 0.5
        min1_y = y + np.cos(-np.pi / 180 * (pos + 90)) * min * 0.5
        min2_x = x + np.sin(-np.pi / 180 * (pos + 90)) * min * 0.5
        min2_y = y - np.cos(-np.pi / 180 * (pos + 90)) * min * 0.5

        line1=self.ax.plot([maj1_x * scale, maj2_x * scale], [maj1_y * scale, maj2_y * scale], color=self.component_color, lw=0.5)
        line2=self.ax.plot([min1_x * scale, min2_x * scale], [min1_y * scale, min2_y * scale], color=self.component_color, lw=0.5)


        return [ellipse,line1,line2]


    def change_plot_lim(self,x_min,x_max,y_min,y_max):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def plotEvpa(self,evpa,rotate_evpa,evpa_len,evpa_distance,levs1_linpol,levs1_i):

        evpa_len=evpa_len*self.clean_image.degpp*self.clean_image.scale

        stokes_i=self.clean_image.Z
        # plot EVPA
        evpa = evpa + rotate_evpa / 180 * np.pi

        # create mask where to plot EVPA (only where stokes i and lin pol have plotted contours)
        mask = np.zeros(np.shape(stokes_i), dtype=bool)
        mask[:] = (self.clean_image.lin_pol > levs1_linpol[0]) * (stokes_i > levs1_i[0])
        YLoc, XLoc = np.where(mask)

        y_evpa = evpa_len * np.cos(evpa[mask])
        x_evpa = evpa_len * np.sin(evpa[mask])
        evpa=evpa[mask]

        SelPix = range(0, len(stokes_i), int(evpa_distance))

        lines = []
        for i in range(0, len(XLoc)):
            if XLoc[i] in SelPix and YLoc[i] in SelPix:
                Xpos = float(self.clean_image.X[XLoc[i]])
                Ypos = float(self.clean_image.Y[YLoc[i]])
                Y0 = float(Ypos - y_evpa[i] / 2.)
                Y1 = float(Ypos + y_evpa[i] / 2.)
                X0 = float(Xpos - x_evpa[i] / 2.)
                X1 = float(Xpos + x_evpa[i] / 2.)
                lines.append(((X0, Y0), (X1, Y1)))
        lines = tuple(lines)


        # plot the evpas
        evpa_lines = LineCollection(lines, colors=self.evpa_color, linewidths=self.evpa_width)
        self.ax.add_collection(evpa_lines)


def get_sigma_levs(image,  # 2d array/list
                   sigma_contour_limit=3  # choose the lowest sigma contour to plot
                   ):
    """Takes an image (2d) array as input and calculates the sigma levels for plotting using a histogram approach

    Args:
        image: 2d array of image data
        sigma_contour_limit: Sigma level for the lowest sigma contour to plot

    Returns:
        Positive and negative sigma levels of the image (the lowest one corresponds to the sigma_contour_limit
    """

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
    levs1 = t.mean.value + np.min(Z1) - 10 ** (-5) + sigma_contour_limit * t.stddev.value * np.logspace(0, 100, 100,
                                                                                                        endpoint=False,
                                                                                                        base=2)
    levs = t.mean.value + np.min(Z1) - 10 ** (-5) - sigma_contour_limit * t.stddev.value * np.logspace(0, 100, 100,
                                                                                                       endpoint=False,
                                                                                                       base=2)
    levs = np.flip(levs)
    levs = np.concatenate((levs, levs1))

    return levs, levs1

#gets components from .fits file
def getComponentInfo(filename):
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
    return data_df

def write_mod_file(model_df,writepath,freq,scale=60*60*1000):
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

    for ind in range(len(flux)):
        print(" "+"{:.8f}".format(flux[ind])+"   "+
              "{:.8f}".format(radius[ind])+"    "+
              "{:.3f}".format(theta[ind])+"   "+
              "{:.7f}".format(maj[ind]*scale)+"    "+
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
def total_flux_from_mod(mod_file):
    """needs a mod_file as input an returns the total flux
    Args:
        mod_file: Path to a .mod file
    Returns:
        The total flux in the .mod file (usually in mJy, depending on the .mod file)
    """

    lines=open(mod_file).readlines()
    total_flux=0
    for line in lines:
        if not line.startswith("!"):
            linepart=line.split()
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

    return np.average(data[x_min:x_max,y_min:y_max]) #TODO check order of x/y here and if AVERAGE is the correct thing to do!!!

