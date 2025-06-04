from os import write
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import models, fitting
import os
from astropy.time import Time
import sys
import pexpect
from datetime import datetime
from astropy.time import Time
from scipy.ndimage import fourier_shift, shift
from skimage.draw import disk, ellipse
from skimage.registration import phase_cross_correlation
from scipy.interpolate import RegularGridInterpolator
import copy
from sympy import Ellipse, Point, Line
from astropy.utils.exceptions import ErfaWarning
import numpy as np
import scipy.ndimage
from astropy.nddata import Cutout2D
from astroquery.ipac.ned import Ned
from astropy.modeling import models, fitting
from scipy import integrate
from vcat.ridgeline import Ridgeline
from vcat.plots.fits_image import FitsImage
from vcat.plots.jet_profile_plot import JetProfilePlot
from vcat.kinematics import Component
from vcat.helpers import *
from vcat.stacking_helpers import fold_with_beam, modelfit_difmap
from skimage.measure import profile_line
import warnings

#initialize logger
from vcat.config import logger,difmap_path,uvw,mfit_err_method,res_lim_method,noise_method, plot_colors, plot_markers
warnings.simplefilter('ignore', ErfaWarning)

class ImageData(object):

    """ Class to handle VLBI Image data (single image with or without polarization at one frequency)

    Attributes:
        General:

            name (str): Source name of the observation
            date (str): Date of the observation
            mjd (float): MJD of the observation
            freq (float): Frequency of the observation in Hz
            beam_maj (float): Beam Major Axis in the intrinsic image scale (usually 'mas')
            beam_min (float): Beam Minor Axis in the intrinsic image scale (usually 'mas')
            beam_pa (float): Beam position angle in degrees (North through East)
            scale (float): Conversion from degrees to the intrinsic image scale (for 'mas': 3.6e6)
            degpp (float): Degrees per pixel
            unit (str): Intrinsic Scale Unit of the image ('mas', 'arcsec', 'arcsec', 'deg')

        Maps:
            stokes_i (list[list[float]]): 2d-array of the Stokes I image
            stokes_q (list[list[float]]): 2d-array of the Stokes Q image (if polarization loaded)
            stokes_u (list[list[float]]): 2d-array of the Stokes U image (if polarization loaded)
            residual_map (list[list[float]]): 2d-array of the residual map (if .uvf file provided)
            lin_pol (list[list[float]]): 2d-array of the linear polarization
            evpa (list[list[float]]): 2d-array of the EVPA
            mask (list[list[bool]]): Image mask

        Model:
            model (DataFrame): DataFrame with all components of the loaded model
            model_i (DataFrame): DataFrame with all Stokes I clean components
            model_q (DataFrame): DataFrame with all Stokes Q clean components
            model_u (DataFrame): DataFrame with all Stokes U clean components
            components (list[Component]): List of Modelfit-Components
        Image Parameters:
            noise (float): Image noise in Jy, calculated using the specified 'noise_method'
            pol_noise (float): Image noise of the linear polarization image in Jy
            noise_3sigma (float): 3-sigma Image noise level in Jy
            pol_noise_3sigma (float): 3-sigma Polarization noise level in Jy
            integrated_flux_image (float): Integrated flux density of the entire image (pixel sum)
            integrated_flux_clean (float): Integrated flux density from the Stokes I clean model
            integrated_pol_flux_image (float): Integrated linearly polarized flux density of the entire image (pixel sum)
            integrated_pol_flux_clean (float): Integrated linearly polarized flux density from Stokes Q and U clean models
            evpa_average (float): Average EVPA calculated from Stokes Q and U clean models (in rad!).
            frac_pol (float): Fractional polarization of the image (integrated_flux_pol_clean/integrated_flux_clean)
        Ridgelines:
            ridgeline (Ridgeline): Ridgeline of the image (can be created with self.get_ridgeline())
            counter_ridgeline (Ridgline): Counter-Ridgeline of the image (can be created with self.get_ridgeline())
        Filepaths:
            file_path (str): File path to Stokes I .fits file
            model_file_path (str): File path to modelfit .fits file
            stokes_q_path (str): File path to Stokes Q .fits file
            stokes_u_path (str): File path to Stokes U .fits file
            stokes_i_mod_file (str): File path to Stokes I clean model .mod file
            stokes_q_mod_file (str): File path to Stokes Q clean model .mod file
            stokes_u_mod_file (str): File path to Stokes U clean model .mod file
            model_mod_file (str): File path to the modelfit .mod file
            residual_map_path (str): Path to the .fits file of the residual map (if .uvf file provided)
        Additional Maps:
            spix (list[list[float]]): 2d-array of spectral index data (if loaded)
            rm (list[list[float]]): 2d-array of rotation measure data (if loaded)
            turnover (list[list[float]]): 2d-array of turnover frequency data (if loaded)
            turnover_flux (list[list[float]]): 2d-array of turnover flux density data (if loaded)
            turnover_error (list[list[float]]): 2d-array of turnover frequency error data (if loaded)
            turnover_chi_sq (list[list[float]]): 2d-array of turnover-fit chi-squared values
    """

    def __init__(self,
                 fits_file="",
                 uvf_file="",
                 stokes_i=[],
                 model="",
                 lin_pol=[],
                 evpa=[],
                 pol_from_stokes=True,
                 mask="",
                 ridgeline="",
                 counter_ridgeline="",
                 stokes_q="",
                 stokes_u="",
                 comp_ids=[],
                 auto_identify=True,
                 core_comp_id=0,
                 redshift=0,
                 query_redshift=True,
                 M=0,
                 model_save_dir="tmp/",
                 is_casa_model=False,
                 noise_method=noise_method, #choose noise method
                 mfit_err_method=mfit_err_method,
                 res_lim_method=res_lim_method,
                 correct_rician_bias=False,
                 error=0.05, #relative error flux densities,
                 fit_comp_polarization=False,
                 difmap_path=difmap_path):

        """
        Initializes an ImageData object to handle a full-polarization VLBI data set at one epoch and one frequency.

        Args:
            fits_file (str): Input .fits file(s) (Stokes I or full polarization, e.g. from CASA)
            uvf_file (str): Input .uvf file(s)
            stokes_i (list[list[float]]): Input of Stokes-I data as a 2d-array
            model (str): Input of modelfit .fits or .mod file (e.g., from DIFMAP), for CASA .fits model, set is_casa_model=True
            lin_pol (list[list[float]]): 2d array of linear polarized intensity values (if using, set pol_from_stokes=False)
            evpa (list[list[float]]): 2d array of Electric Vector Position Angle (EVPA) (if using, set pol_from_stokes=False)
            pol_from_stokes (bool): Choose whether to import data from fits-files or from lin_pol/evpa
            mask (list[list[bool]]): 2d-array of an image mask
            ridgeline (Ridgeline): Ridgeline of the image
            counter_ridgeline (Ridgeline: Counter ridgeline of the image.
            stokes_q (str or list[list[float]]): Input Stokes-Q .fits file or 2d array of Stokes-Q image
            stokes_u (str or list[list[float]]): Input Stokes-U .fits file or 2d array of Stokes-U image
            comp_ids (list[int]): list of integers to assign as component number (from top to bottom .mod file or .fits header)
            auto_identify (bool): If true and no comp_ids provided components will automatically be named
            core_comp_id (int): Component ID of the core component
            redshift (float): Redshift of the source
            query_redshift (bool): Choose whether to query redshift automatically from NED
            M (float): Black hole mass
            model_save_dir (str): Directory where temporary data for VCAT operations will be stored
            is_casa_model (bool): If using a CASA .fits model for 'model', set to True
            noise_method (str): Choose method to calculate image noise ('Histogram Fit', 'box', 'Image RMS', 'DIFMAP')
            mfit_err_method (str): Choose method to compute modelcomponent errors ('flat', 'Schinzel12', 'Weaver22')
            res_lim_method (str): Choose method to compute component resolution limit ('Kovalev05', 'Lobanov05','beam')
            correct_rician_bias (bool): Choose whether to correct polarization for Rician Bias
            error (float): Set relative error on the flux density scale
            fit_comp_polarization (bool): Choose whether to fit polarization of modelfit components
            difmap_path (str): Path to the folder of your DIFMAP installation
        """

        if model=="":
            self.model_inp=False
        else:
            if fits_file=="":
                fits_file=model
            self.model_inp=True
        self.file_path = fits_file
        self.fits_file = fits_file
        self.lin_pol=lin_pol
        self.evpa=evpa
        self.stokes_i=stokes_i
        self.uvf_file=uvf_file
        self.difmap_path=difmap_path
        self.residual_map_path=""
        self.residual_map = []
        self.noise_method=noise_method
        self.is_casa_model=is_casa_model
        self.model_save_dir=model_save_dir
        self.correct_rician_bias=correct_rician_bias
        self.fit_comp_pol = fit_comp_polarization
        self.error=error
        self.M=M
        if ridgeline=="":
            self.ridgeline=Ridgeline()
        else:
            self.ridgeline=ridgeline
        if counter_ridgeline=="":
            self.counter_ridgeline=Ridgeline()
        else:
            self.counter_ridgeline=counter_ridgeline


        if fits_file=="":
            #if no fits file was loaded try to get the dirty image
            if uvf_file!="":
                #get dirty map from uvf file
                get_residual_map(uvf_file, "", difmap_path=difmap_path, channel="i",
                                 save_location="/tmp/dirty_image.fits", npix=2048, pxsize=0.05,
                                 do_selfcal=False)
                fits_file="/tmp/dirty_image.fits"
                self.fits_file=fits_file
                self.file_path=fits_file
            else:
                self.no_fits=True

        # Read clean files in
        if fits_file!="":
            hdu_list=fits.open(fits_file)
            self.hdu_list = hdu_list
            self.no_fits=False

        
        self.stokes_q_path=stokes_q
        self.stokes_u_path=stokes_u
        stokes_q_path=stokes_q
        stokes_u_path=stokes_u
        #read stokes data from input files if defined
        if stokes_q != "":
            try:
                q_fits=fits.open(stokes_q)
                try:
                    stokes_q = q_fits[0].data[0, 0, :, :]
                except:
                    stokes_q = q_fits[0].data
                q_fits.close()
            except:
                stokes_q=stokes_q
        else:
            stokes_q=[]

        if stokes_u != "":
            try:
                u_fits=fits.open(stokes_u)
                try:
                    stokes_u = u_fits[0].data[0, 0, :, :]
                except:
                    stokes_u = u_fits[0].data
                u_fits.close()
            except:
                stokes_u = stokes_u
        else:
            stokes_u=[]

        self.stokes_u=stokes_u
        self.stokes_q=stokes_q

        # Set name
        self.name = hdu_list[0].header["OBJECT"]
        self.date = get_date(fits_file)
        self.mjd = Time(self.date).mjd
        try:
            self.freq = float(hdu_list[0].header["CRVAL3"])  # frequency in Hertz
        except:
            try:
                self.freq = float(hdu_list[0].header["FREQ"])
            except:
                self.freq = 15000000000


        #get redshift
        if redshift==0 and query_redshift:
            try:
                self.redshift = np.average(Ned.get_table(self.name, table="redshifts")["Published Redshift"])
                logger.debug(f"Redshift for {self.name} automatically determined from NED: {self.redshift}")
            except:
                self.redshift = 0.00
        else:
            self.redshift=redshift

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
        # FMP suggestion: add microarcseconds for possible scale

        # Set beam parameters
        try:
            # DIFMAP style
            self.beam_maj = hdu_list[0].header["BMAJ"] * self.scale
            self.beam_min = hdu_list[0].header["BMIN"] * self.scale
            self.beam_pa = hdu_list[0].header["BPA"]
        except:
            try:
                # TODO check if this is actually working!
                # CASA style
                self.beam_maj, self.beam_min, self.beam_pa, na, nb = hdu_list[1].data[0]
                self.beam_maj = self.beam_maj * 1000  # convert to mas
                self.beam_min = self.beam_min * 1000  # convert to mas
            except:
                logger.warning("No input beam information!")
                self.beam_maj = 0
                self.beam_min = 0
                self.beam_pa = 0


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
            try:
                self.Z = self.image_data[0, 0, :, :]
            except:
                self.Z = self.image_data

        else:
            try:
                self.Z=self.stokes_i
            except:
                pass


        #handle model loading
        self.model_file_path = model
        if self.model_file_path=="":
            self.model_file_path=self.fits_file
        elif not isinstance(model, pd.DataFrame) and not is_fits_file(model): #Careful, this may not work for CASA style .fits files!
            #this means it is a .mod file -> will create .fits file from it
            os.makedirs(model_save_dir + "mod_files_model/", exist_ok=True)
            new_model_fits=model_save_dir+"mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz"
            if difmap_path!="" and uvf_file!="":
                # use difmap to load the model and create model .fits file and store it as model_file_path
                fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                               bmaj=self.beam_maj, bmin=self.beam_min, posa=self.beam_pa,
                               outname=new_model_fits, n_pixel=len(self.X)*2, pixel_size=self.degpp*self.scale,
                               mod_files=[model],clean_mod_files=[model], uvf_files=[uvf_file], do_selfcal=True)

            else:
                #copy the clean .fits file and write the model info to the header and store it as model_file_path
                #get model first:
                model_df = getComponentInfo(model,scale=self.scale)

                #now modify fits file
                f=fits.open(self.fits_file)
                # FITS column names
                fits_columns = f[1].data.names

                # Manually map DataFrame columns to FITS structure
                column_mapping = {
                    "FLUX": "Flux",
                    "DELTAX": "Delta_x",
                    "DELTAY": "Delta_y",
                    "MAJOR AX": "Major_axis",
                    "MINOR AX": "Minor_axis",
                    "POSANGLE": "PA",
                    "TYPE OBJ": "Typ_obj",
                }

                # Ensure correct order and match dtype
                new_data_array = np.array(
                    [tuple(df[column_mapping[col]] for col in fits_columns) for _, df in model_df.iterrows()],
                    dtype=f[1].data.dtype  # Ensure the same dtype as the original FITS table
                )

                # Overwrite the FITS table with the new structured array
                f[1].data = new_data_array
                f[1].header['XTENSION'] = 'BINTABLE'
                f.writeto(new_model_fits+".fits",overwrite=True)
                f.close()

            self.model_file_path = new_model_fits + ".fits"
            model = self.model_file_path

        #overwrite fits image data with stokes_i input if given
        if not stokes_i==[]:
            self.Z=stokes_i

        #read in polarization input

        # check if FITS file contains more than just Stokes I
        self.only_stokes_i = False
        if hdu_list[0].data.shape[0] == 1:
            self.only_stokes_i = True
        elif len(hdu_list[0].data.shape) == 2:
            self.only_stokes_i = True
        if (np.shape(self.Z) == np.shape(stokes_q) and np.shape(self.Z) == np.shape(stokes_u) and
                        np.shape(stokes_q) == np.shape(stokes_u)):
            self.only_stokes_i = True #in this case override the polarization data with the data that was input to Q and U

        if self.only_stokes_i:
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
            try:
                self.image_data[0, 0, :, :] = self.Z
            except:
                self.image_data = self.Z
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

        try:
            self.difmap_noise = float(hdu_list[0].header["NOISE"])
        except:
            self.difmap_noise = 0
        

        try:
            q_fits=fits.open(stokes_q_path)
            u_fits=fits.open(stokes_u_path)
            self.difmap_pol_noise = np.sqrt(float(q_fits[0].header["NOISE"])**2+float(u_fits[0].header["NOISE"])**2)
            q_fits.close()
            u_fits.close()
        except:
            self.difmap_pol_noise = 0
    
        #calculate image noise according to the method selected
        logger.debug("Calculating Stokes I noise")
        unused, levs_i = get_sigma_levs(self.Z, 1,noise_method=self.noise_method,noise=self.difmap_noise) #get noise for stokes i

        if np.sum(self.lin_pol)!=0:
            logger.debug("Calculating Pol noise")
            unused, levs_pol = get_sigma_levs(self.lin_pol, 1,noise_method=self.noise_method,noise=self.difmap_noise) #get noise for polarization
        else:
            levs_pol=[0]

        self.noise = levs_i[0]
        self.pol_noise = levs_pol[0]

        #calculate integrated total flux in image
        self.integrated_flux_image = JyPerBeam2Jy(np.sum(self.Z), self.beam_maj, self.beam_min, self.degpp * self.scale)

        #calculate integrated pol flux in image
        self.integrated_pol_flux_image = JyPerBeam2Jy(np.sum(self.lin_pol),self.beam_maj,self.beam_min,self.degpp*self.scale)

        if not is_casa_model:
            try:
                #TODO basic checks if file is valid
                self.model=getComponentInfo(self.model_file_path, scale=self.scale)
                #write .mod file from .fits input
                os.makedirs(model_save_dir,exist_ok=True)
                os.makedirs(model_save_dir+"mod_files_model/",exist_ok=True)
                if self.model is not None:
                    self.model_mod_file=model_save_dir+"mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
                    write_mod_file(self.model, self.model_mod_file, freq=self.freq)
            except:
                logger.warning("FITS file does not contain model extension!")
        else:
            #TODO basic checks if file is valid
            os.makedirs(model_save_dir,exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_clean", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_q", exist_ok=True)
            os.makedirs(model_save_dir + "mod_files_u", exist_ok=True)
            self.stokes_i_mod_file=model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            self.write_mod_file_from_casa(channel="i", export=self.stokes_i_mod_file)
            self.stokes_q_mod_file=model_save_dir+"mod_files_q/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            self.write_mod_file_from_casa(channel="q", export=self.stokes_q_mod_file)
            self.stokes_u_mod_file=model_save_dir+"mod_files_u/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            self.write_mod_file_from_casa(channel="u", export=self.stokes_u_mod_file)
        try:
            os.makedirs(model_save_dir+"mod_files_clean", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_q", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_u", exist_ok=True)
            #try to import model which is attached to the main .fits file
            model_i = getComponentInfo(fits_file, scale=self.scale)
            self.model_i = model_i
            self.stokes_i_mod_file=model_save_dir+"mod_files_clean/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(model_i, self.stokes_i_mod_file, freq=self.freq)
            #load stokes q and u clean models
            self.model_q=getComponentInfo(stokes_q_path, scale=self.scale)
            self.stokes_q_mod_file=model_save_dir+"mod_files_q/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(self.model_q, self.stokes_q_mod_file, freq=self.freq)
            self.model_u=getComponentInfo(stokes_u_path, scale=self.scale)
            self.stokes_u_mod_file=model_save_dir+"mod_files_u/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(self.model_u, self.stokes_u_mod_file, freq=self.freq)
        except:
            pass

        #calculate residual map if uvf and modelfile present
        if self.uvf_file!="" and self.model_file_path!="" and not is_casa_model and  self.difmap_path!="":
            os.makedirs(model_save_dir+"residual_maps", exist_ok=True)
            self.residual_map_path = model_save_dir + "residual_maps/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq / 1e9).replace(".",
                                                                                                                 "_") + "GHz_residual.fits"
            get_residual_map(self.uvf_file,model_save_dir+ "mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod",
                             difmap_path=self.difmap_path,
                             save_location=self.residual_map_path,npix=len(self.X),pxsize=self.degpp*self.scale)

            self.residual_map=fits.open(self.residual_map_path)[0].data[0,0,:,:]
        
        #save modelfit (or clean) components as Component objects
        self.components=[]

        if self.model_inp:
            #only do this if a model was specified explicitely
            for ind,comp in self.model.reset_index().iterrows():
                #use provided comp_id
                try:
                    comp_id=comp_ids[ind]
                except:
                    #assign automatic comp_id
                    if auto_identify:
                        comp_id=ind
                    else:
                        comp_id=-1

                #check if component is the core component
                if comp_id==core_comp_id:
                    is_core=True
                else:
                    is_core=False

                #calculate component SNR
                if self.uvf_file!="" and self.difmap_path!="":
                    S_p, rms = get_comp_peak_rms(comp["Delta_x"]*self.scale,comp["Delta_y"]*self.scale,
                                                 self.fits_file,self.uvf_file,self.model_mod_file,
                                                 weighting=uvw, difmap_path=self.difmap_path)
                    comp_snr = S_p/rms
                else:
                    if ind == 0:
                        logger.warning('No .uvfits file or difmap path provided. Calculating modelfit component SNR based on the clean map only.')
                    # TODO: use .fits file from Gaussian modelfit instead of clean map
                    S_p = self.get_pixel_value(comp["Delta_x"]*self.scale,
                                                   comp["Delta_y"]*self.scale)
                    rms=self.noise
                    comp_snr = S_p/rms

                component=Component(comp["Delta_x"],comp["Delta_y"],comp["Major_axis"],comp["Minor_axis"],
                                    comp["PA"],comp["Flux"],self.date,self.mjd,Time(self.mjd,format="mjd").decimalyear,component_number=comp_id,
                                    redshift=redshift, is_core=is_core,beam_maj=self.beam_maj,beam_min=self.beam_min,beam_pa=self.beam_pa,
                                    freq=self.freq,noise=rms, scale=self.scale, snr=comp_snr,error_method=mfit_err_method,res_lim_method=res_lim_method)
                self.components.append(component)

            #set core
            self.set_core_component(core_comp_id)
            if self.uvf_file!="" and fit_comp_polarization:
                logger.debug("Retrieving polarization information for modelfit components.")
                self.fit_comp_polarization()
            else:
                if fit_comp_polarization:
                    logger.warning("Trying to fit component polarization, but no uvf file loaded!")
                else:
                    logger.debug("Not fitting component polarization")
        
        hdu_list.close()

        #calculate cleaned flux density from mod files
        #first stokes I
        try:
            self.integrated_flux_clean=total_flux_from_mod(self.model_save_dir+"mod_files_clean/"  + self.name + "_" +
                                                           self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
        except:
            self.integrated_flux_clean = 0
        #and then polarization
        try:
            flux_q=total_flux_from_mod(self.model_save_dir+"mod_files_q/" + self.name + "_" + self.date + "_" +
                                       "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            flux_u=total_flux_from_mod(self.model_save_dir+"mod_files_u/" + self.name + "_" + self.date + "_" +
                                       "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            self.integrated_pol_flux_clean=np.sqrt(flux_u**2+flux_q**2)
            self.frac_pol = self.integrated_pol_flux_clean / self.integrated_flux_clean
            self.evpa_average = 0.5*np.arctan2(flux_u,flux_q)
        except:
            self.integrated_pol_flux_clean=0
            self.frac_pol = 0

        #correct rician bias
        if correct_rician_bias:
            lin_pol_sqr = (self.lin_pol ** 2 - self.pol_noise ** 2)
            lin_pol_sqr[lin_pol_sqr < 0.0] = 0.0
            self.lin_pol = np.sqrt(lin_pol_sqr)

        # initialize mask
        if len(mask)==0:
            self.mask = np.zeros_like(self.Z, dtype=bool)
            #test masking
            #self.mask[0:200]=np.ones_like(self.Z[0:200],dtype=bool)
            #self.masking(mask_type="cut_left",args=-200)
            #set mask where Image is None
            self.mask[np.isnan(self.Z)]=True
        else:
            if np.shape(mask) != np.shape(self.Z):
                print(np.shape(mask),np.shape(self.Z))
                logger.warning("Mask input format invalid, Mask reset to no mask.")
                self.mask = np.zeros_like(self.Z, dtype=bool)
            else:
                self.mask=mask

        # additional parameters only used for spectral index type data
        self.is_spix=False
        self.spix=[]
        self.spix_vmin=-3
        self.spix_vmax=5

        #additional parameter only used for rotation measure data
        self.is_rm=False
        self.rm=[]
        self.rm_vmin=""
        self.rm_vmax=""

        # additional parameter only used for Spectral turnover data
        self.is_turnover = False
        self.turnover = []
        self.turnover_flux = []
        self.turnover_error = []
        self.turnover_chi_sq = []

    #print function for ImageData
    def __str__(self):
        output=["\n"]
        try:
            freq_ghz="{:.1f}".format(self.freq*1e-9)
            output.append(f"Image of the source {self.name} at frequency {freq_ghz} GHz on {self.date} \n")
            output.append(f"    Total cleaned flux: {self.integrated_flux_clean*1000:.3f} mJy \n")
            output.append(f"    Image Noise: {self.noise*1000:.3f} mJy using method '{self.noise_method}'\n")

            #polarization info
            if np.sum(self.lin_pol)!=0 and np.sum(self.evpa)!=0:
                #print polarization info if pol data was loaded
                output.append("Polarization information:\n")
                output.append(f"    Pol Flux: {self.integrated_pol_flux_clean*1000:.3f} mJy ({self.frac_pol*100:.2f}%)\n")
                output.append(f"    Pol Noise: {self.pol_noise*1000:.3f} mJy using method '{self.noise_method}'\n")
                output.append(f"    Average EVPA direction: {self.evpa_average/np.pi*180:.2f}°\n")
            else:
                output.append("No polarization data loaded.\n")

            #model info
            if self.model_file_path!=self.fits_file:
                output.append("Model information: \n")
            else:
                output.append("No model loaded. Clean model info: \n")
            model_flux = total_flux_from_mod(self.model_mod_file)
            num_comps = len(self.model)
            output.append(f"    Model Flux: {model_flux*1000:.3f} mJy \n")
            output.append(f"    Number of Components: {num_comps}")

            return "".join(output)
        except:
            return "No data loaded yet."

    def write_mod_file_from_casa(self,channel="i",export="export.mod"):

        """Writes a .mod file from a CASA exported .fits model file.
            Args:
                file_path: File path to a .fits model file as exported from a CASA .model file (e.g. with exportfits() in CASA)
                channel: Choose the Stokes channel to use (options: "i","q","u","v")
                export: File path where to write the .mod file
            Returns:
                Nothing, but writes a .mod file to export
            """

        if channel == "i":
            clean_map = self.Z
        elif channel == "q":
            clean_map = self.stokes_q
        elif channel == "u":
            clean_map = self.stokes_u
        else:
            raise Exception("Please enter a valid channel (i,q,u)")

        # read out clean components from pixel map
        delta_x = []
        delta_y = []
        flux = []
        zeros = []
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                if clean_map[j][i] > 0:
                    delta_x.append(self.X[i] / self.scale)
                    delta_y.append(self.Y[j] / self.scale)
                    flux.append(clean_map[j][i])
                    zeros.append(0.0)

        # create model_df
        model_df = pd.DataFrame(
            {'Flux': flux,
             'Delta_x': delta_x,
             'Delta_y': delta_y,
             'Major_axis': zeros,
             'Minor_axis': zeros,
             'PA': zeros,
             'Typ_obj': zeros
             })

        # create mod file
        write_mod_file(model_df, export, self.freq, self.scale)

    def get_pixel_value(self,x,y):
        Xind=closest_index(self.X,x)
        Yind=closest_index(self.Y,y)

        return self.Z[Yind,Xind]

    def copy(self):
        return copy.copy(self)

    def export(self,outputfile,polarization="I"):
        """
        Function to export fits file

        Args:
            outputfile (str): Name/path of the intended output file
            polarization (str): Polarization to export ('I','Q','U')
        """
        if polarization=="I":
            os.system(f"cp {self.file_path} {outputfile}")
            logger.info(f"Stokes {polarization} succesfully exported to {outputfile}.")
        elif polarization=="Q":
            if self.stokes_q_path=="":
                logger.info(f"Stokes {polarization} succesfully exported to {outputfile}.")
            else:
                os.system(f"cp {self.stokes_q_path} {outputfile}")
                logger.info(f"Stokes {polarization} succesfully exported to {outputfile}.")
        elif polarization=="U":
            if self.stokes_u_path=="":
                logger.info(f"Stokes {polarization} succesfully exported to {outputfile}.")
            else:
                os.system(f"cp {self.stokes_u_path} {outputfile}")
                logger.info(f"Stokes {polarization} succesfully exported to {outputfile}.")

    def regrid(self,npix="",pixel_size="",weighting=uvw,useDIFMAP=True,mask_outside=False):
        """
        This method regrids the image in full polarization

        Args:
            npix (int): Number of pixels in ONE direction
            pixel_size (float): Size of pixel in image scale units (usually mas)
            weighting (list[int]): DIFMAP style weighting
            useDIFMAP (bool): Choose whether to regrid using DIFMAP or not
            mask_outside (bool): Choose whether new image ares created through regridding will be masked automatically (bool)
        Returns:
            regridded ImageData object
        """
        logger.debug("Regridding Image")

        if len(self.X)==npix and len(self.Y)==npix and pixel_size==self.degpp*self.scale:
            return self

        n2 = len(self.X)
        n1 = len(self.Y)

        # Original grid (centered)
        x_old = (np.arange(n2) - (n2 - 1) / 2) * self.degpp * self.scale
        y_old = (np.arange(n1) - (n1 - 1) / 2) * self.degpp * self.scale

        # New grid (centered)
        x_new = (np.arange(npix) - (npix - 1) / 2) * pixel_size
        y_new = (np.arange(npix) - (npix - 1) / 2) * pixel_size

        # Generate new grid coordinates
        X_new, Y_new = np.meshgrid(x_new, y_new)
        points = np.array([Y_new.ravel(), X_new.ravel()]).T

        # define interpolator
        def interpolator(image,fill_value=0):
            interpolator = RegularGridInterpolator((y_old, x_old), image, method='linear', bounds_error=False,
                                                   fill_value=fill_value)
            return interpolator

        # regrid mask
        if mask_outside==True:
            fill_value=1
        else:
            fill_value=0


        new_mask = interpolator(self.mask, fill_value)(points).reshape(npix, npix)  # flags new points automatically
        new_mask[new_mask < 0.5] = False
        new_mask[new_mask >= 0.5] = True

        if self.uvf_file=="" or useDIFMAP==False:
            # Interpolate values at new grid points
            new_image_i = interpolator(self.Z)(points).reshape(npix, npix)

            #try polarization
            try:
                new_image_q = interpolator(self.stokes_q)(points).reshape(npix, npix)
                new_image_u = interpolator(self.stokes_u)(points).reshape(npix, npix)
            except:
                logger.warning("Unable to regrid polarization, probably no polarization loaded")


            # write outputs to the fits files
            if self.only_stokes_i:
                # this means DIFMAP style fits image
                with fits.open(self.fits_file) as f:
                    #overwrite image data
                    f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                    f[0].data[0, 0, :, :] = new_image_i
                    new_stokes_i_fits = self.model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                    try:
                        f[1].header['XTENSION'] = 'BINTABLE' #This is a bug fix that is needed for some .fits files, otherwise writeto throws an error
                    except:
                        pass
                    #modify header parameters to new npix and pixelsize
                    f[0].header["NAXIS1"]=npix
                    f[0].header["NAXIS2"]=npix
                    f[0].header["CDELT1"]=-pixel_size/self.scale
                    f[0].header["CDELT2"]=pixel_size/self.scale
                    f[0].header["CRPIX1"]=int(f[0].header["CRPIX1"]/len(self.X)*npix)
                    f[0].header["CRPIX2"]=int(f[0].header["CRPIX2"]/len(self.X)*npix)
                    f.writeto(new_stokes_i_fits, overwrite=True)

                if len(self.stokes_q) > 0:
                    with fits.open(self.stokes_q_path) as f:
                        # overwrite image data
                        f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                        f[0].data[0, 0, :, :] = new_image_q
                        new_stokes_q_fits = self.model_save_dir+"mod_files_q/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'  # This is a bug fix that is needed for some .fits files, otherwise writeto throws an error
                        except:
                            pass
                        # modify header parameters to new npix and pixelsize
                        f[0].header["NAXIS1"] = npix
                        f[0].header["NAXIS2"] = npix
                        f[0].header["CDELT1"] = -pixel_size / self.scale
                        f[0].header["CDELT2"] = pixel_size / self.scale
                        f[0].header["CRPIX1"] = int(f[0].header["CRPIX1"] / len(self.X) * npix)
                        f[0].header["CRPIX2"] = int(f[0].header["CRPIX2"] / len(self.X) * npix)
                        f.writeto(new_stokes_q_fits, overwrite=True)
                else:
                    new_stokes_q_fits=""


                if len(self.stokes_u) > 0:
                    with fits.open(self.stokes_u_path) as f:
                        # overwrite image data
                        f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                        f[0].data[0, 0, :, :] = new_image_u
                        new_stokes_u_fits = self.model_save_dir+"mod_files_u/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'  # This is a bug fix that is needed for some .fits files, otherwise writeto throws an error
                        except:
                            pass
                        # modify header parameters to new npix and
                        # pixelsize
                        f[0].header["NAXIS1"] = npix
                        f[0].header["NAXIS2"] = npix
                        f[0].header["CDELT1"] = -pixel_size / self.scale
                        f[0].header["CDELT2"] = pixel_size / self.scale
                        f[0].header["CRPIX1"] = int(f[0].header["CRPIX1"] / len(self.X) * npix)
                        f[0].header["CRPIX2"] = int(f[0].header["CRPIX2"] / len(self.X) * npix)
                        f.writeto(new_stokes_u_fits, overwrite=True)
                else:
                    new_stokes_u_fits = ""

            else:
                # CASA style
                f = fits.open(self.fits_file)
                # overwrite image data
                f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                f[0].data[0, 0, :, :] = new_image_i
                f[0].data[1, 0, :, :] = new_image_q
                f[0].data[2, 0, :, :] = new_image_u
                f[0].header["NAXIS1"] = npix
                f[0].header["NAXIS2"] = npix
                f[0].header["CDELT1"] = -pixel_size / self.scale
                f[0].header["CDELT2"] = pixel_size / self.scale
                f[0].header["CRPIX1"] = int(f[0].header["CRPIX1"] / len(self.X) * npix)
                f[0].header["CRPIX2"] = int(f[0].header["CRPIX2"] / len(self.X) * npix)
                new_stokes_i_fits = self.model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                f.writeto(new_stokes_i_fits, overwrite=True, output_verify='ignore')
                new_stokes_q_fits=""
                new_stokes_u_fits=""

            #if model loaded try regridding as well
            try:
                if not self.model_file_path == self.fits_file:
                    if not self.model_file_path=="":
                        with fits.open(self.model_file_path) as f:
                            new_image_model = interpolator(f[0].data[0, 0, :, :])(points).reshape(npix,npix)
                            f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                            f[0].data[0, 0, :, :] = new_image_model
                            new_model_fits = self.model_save_dir + "mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                    self.freq / 1e9).replace(".", "_") + "GHz.fits"
                            try:
                                f[1].header['XTENSION'] = 'BINTABLE'  # This is a bug fix that is needed for some .fits files, otherwise writeto throws an error
                            except:
                                pass
                            f[0].header["NAXIS1"] = npix
                            f[0].header["NAXIS2"] = npix
                            f[0].header["CDELT1"] = -pixel_size / self.scale
                            f[0].header["CDELT2"] = pixel_size / self.scale
                            f[0].header["CRPIX1"]=int(f[0].header["CRPIX1"]/len(self.X)*npix)
                            f[0].header["CRPIX2"]=int(f[0].header["CRPIX2"]/len(self.X)*npix)
                            f.writeto(new_model_fits, overwrite=True)
                    else:
                        new_model_fits=""
                else:
                    new_model_fits=new_stokes_i_fits
            except:
                logger.warning("Model not regridded, probably no model loaded.")
                new_model_fits=""

        else:
            npix=npix*2 #DIFMAP npix convention
            #Using DIFMAP
            # restore Stokes I
            new_stokes_i_fits = self.stokes_i_mod_file.replace(".mod", "")

            fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                           bmaj=self.beam_maj, bmin=self.beam_min, posa=self.beam_pa, shift_x=0, shift_y=0,
                           channel="i", output_dir=self.model_save_dir + "mod_files_clean", outname=new_stokes_i_fits,
                           n_pixel=npix, pixel_size=pixel_size,
                           mod_files=[self.stokes_i_mod_file],clean_mod_files=[self.stokes_i_mod_file], uvf_files=[self.uvf_file], weighting=weighting)

            new_stokes_i_fits += ".fits"

            # try to restore modelfit if it is there
            try:
                if not self.model_file_path == self.fits_file:
                    new_model_fits = self.model_mod_file.replace(".mod", "")

                    fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                                   bmaj=self.beam_maj, bmin=self.beam_min, posa=self.beam_pa, shift_x=0, shift_y=0,
                                   channel="i", output_dir=self.model_save_dir + "mod_files_model",
                                   outname=new_model_fits,
                                   n_pixel=npix, pixel_size=pixel_size,
                                   mod_files=[self.model_mod_file],clean_mod_files=[self.stokes_i_mod_file], uvf_files=[self.uvf_file], weighting=weighting)

                    new_model_fits += ".fits"
                else:
                    new_model_fits = new_stokes_i_fits
            except:
                new_model_fits = ""

            # try to restore polarization as well if it is there
            try:
                new_stokes_q_fits = self.stokes_q_mod_file.replace(".mod", "")
                new_stokes_u_fits = self.stokes_u_mod_file.replace(".mod", "")

                fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                               bmaj=self.beam_maj, bmin=self.beam_min, posa=self.beam_pa, shift_x=0, shift_y=0,
                               channel="q", output_dir=self.model_save_dir + "mod_files_q", outname=new_stokes_q_fits,
                               n_pixel=npix, pixel_size=pixel_size,
                               mod_files=[self.stokes_q_mod_file],clean_mod_files=[self.stokes_i_mod_file], uvf_files=[self.uvf_file], weighting=weighting)

                new_stokes_q_fits += ".fits"

                fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                               bmaj=self.beam_maj, bmin=self.beam_min, posa=self.beam_pa, shift_x=0, shift_y=0,
                               channel="u", output_dir=self.model_save_dir + "mod_files_u", outname=new_stokes_u_fits,
                               n_pixel=npix, pixel_size=pixel_size,
                               mod_files=[self.stokes_u_mod_file], clean_mod_files=[self.stokes_i_mod_file],uvf_files=[self.uvf_file], weighting=weighting)

                new_stokes_u_fits += ".fits"

            except:
                new_stokes_q_fits = ""
                new_stokes_u_fits = ""

        if not self.model_inp:
            new_model_fits = ""

        return ImageData(fits_file=new_stokes_i_fits,
                         uvf_file=self.uvf_file,
                         stokes_q=new_stokes_q_fits,
                         stokes_u=new_stokes_u_fits,
                         mask=new_mask,
                         ridgeline=self.ridgeline,
                         redshift=self.redshift,
                         counter_ridgeline=self.counter_ridgeline,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=new_model_fits,
                         correct_rician_bias=self.correct_rician_bias,
                         comp_ids=self.get_model_info()[0],
                         core_comp_id=self.get_model_info()[1],
                         difmap_path=self.difmap_path,
                         fit_comp_polarization=self.fit_comp_pol)

    def plot(self,show=True,savefig="",**kwargs):
        defaults = {
            "stokes_i_sigma_cut": 3,
            "plot_mode": "stokes_i",
            "im_colormap": False,
            "contour": True,
            "contour_color": 'grey',
            "contour_cmap": None,
            "contour_alpha": 1,
            "contour_width": 0.5,
            "im_color": '',
            "do_colorbar": False,
            "plot_ridgeline": False,
            "ridgeline_color": "red",
            "plot_counter_ridgeline": False,
            "counter_ridgeline_color": "red",
            "plot_line" : "",
            "line_color" : "black",
            "line_width" : 2,
            "plot_polar": False,
            "plot_beam": True,
            "beam_color": "grey",
            "plot_model": False,
            "component_color": "black",
            "plot_comp_ids": False,
            "plot_clean": False,
            "plot_mask": False,
            "xlim": [],
            "ylim": [],
            "plot_evpa": False,
            "evpa_width": 1.5,
            "evpa_len": -1,
            "lin_pol_sigma_cut": 3,
            "evpa_distance": -1,
            "fractional_evpa_distance": 0.02,
            "rotate_evpa": 0,
            "evpa_color": "white",
            "title": "",
            "background_color": "white",
            "font_size_axis_title": 8,
            "font_size_axis_tick": 6,
            "rcparams": {}
        }

        params = {**defaults, **kwargs}
        plot=FitsImage(self, **params)
        if savefig!="":
            plot.export(savefig)
        if show:
            plt.show()

        return plot

    def align(self,image_data2,masked_shift=True,method="cross_correlation",beam_arg="common", auto_regrid=False,
              useDIFMAP=True,comp_ids="",weight_by_comp_err=True):
        """
        This function aligns the image to a reference image (image_data2).

        Args:
            image_data2 (ImageData): ImageData object of the reference image
            masked_shift (bool): Choose whether to consider the image masks for alignment
            method: Choose alignment method (Options: 'cross_correlation', 'brightest', 'modelcomp')
            beam_arg (str): Choose which common beam to use (Options: 'common', 'max', 'min'), only applied when auto_regrid=True
            auto_regrid (bool): Choose whether to automatically regrid and restore both images to a common beam and image size.
            useDIFMAP (bool): Choose whether to use DIFMAP for image operations or not.
            comp_ids (int or list[int]): Component IDs to use for the alignment in 'modelcomp' mode.
        Returns:
            new ImageData object aligned to image_data2 (possibly also regridded and restored if auto_regrid=True).
        """
        if self==image_data2:
            return self

        if ((self.Z.shape != image_data2.Z.shape) or self.degpp != image_data2.degpp) or auto_regrid:
            if auto_regrid:
                # if this is selected will automatically convolve with common beam and regrid
                logger.info("Automatically regridding image to minimum pixelsize, smallest FOV and common beam")

                #determin common image parameters
                pixel_size=np.min([self.degpp*self.scale,image_data2.degpp*image_data2.scale])
                min_fov=np.min([self.degpp*len(self.X)*self.scale,image_data2.degpp*len(image_data2.X)*self.scale])
                npix=int(min_fov/pixel_size)

                #get common beam
                common_beam=get_common_beam([self.beam_maj,image_data2.beam_maj],
                                            [self.beam_min,image_data2.beam_min],
                                            [self.beam_pa,image_data2.beam_pa],arg=beam_arg)

                #regrid images
                image_self = self.copy()
                # convolve with common beam
                image_self = image_self.regrid(npix, pixel_size, useDIFMAP=useDIFMAP)
                image_self = image_self.restore(common_beam[0], common_beam[1], common_beam[2], useDIFMAP=useDIFMAP)

                # same for image 2
                image_data2 = image_data2.regrid(npix, pixel_size, useDIFMAP=useDIFMAP)
                image_data2 = image_data2.restore(common_beam[0], common_beam[1], common_beam[2], useDIFMAP=useDIFMAP)



            else:
                if not (method=="modelcomp" or method=="model_comp" or method=="model"):
                    logger.warning("Images do not have the same npix and pixelsize, please regrid first or use auto_regrid=True.")
                    return self
                else:
                    image_self=self.copy()
        else:
            image_self=self.copy()

        if method=="cross_correlation" or method=="crosscorrelation":
            if (np.all(image_data2.mask==False) and np.all(image_self.mask==False)) or masked_shift==False:

                shift,error,diffphase = phase_cross_correlation(image_data2.Z,image_self.Z,upsample_factor=100)
                logger.info('will apply shift (x,y): [{} : {}] {}'.format(-shift[1]*image_self.scale*image_self.degpp, shift[0] *image_self.scale*image_self.degpp,self.unit))
                #print('register images new shift (y,x): {} px +- {}'.format(-shift, error))
            else:
                shift, _, _ = phase_cross_correlation(image_data2.Z,image_self.Z,upsample_factor=100,reference_mask=image_data2.mask,moving_mask=image_self.mask)
                logger.info('will apply shift (x,y): [{} : {}] {}'.format(-shift[1]*image_self.scale*image_self.degpp, shift[0]*image_self.scale*image_self.degpp,self.unit))
                #print('register images new shift (y,x): {} px'.format(-shift))

        elif method=="brightest":
            #align images on brightest pixel
            #find brightest pixel of reference image and image
            x_ind,y_ind = np.unravel_index(np.argmax(image_data2.Z), image_data2.Z.shape)
            x_,y_ = np.unravel_index(np.argmax(image_self.Z), image_self.Z.shape)

            shift=[y_ind-y_,x_ind-x_]
            logger.info('will apply shift (x,y): [{} : {}] {}'.format(-shift[1] * image_self.scale * image_self.degpp,
                                                                 shift[0] * image_self.scale * image_self.degpp,self.unit))
        elif method=="modelcomp" or method=="model_comp" or method=="model":
            #get models of both images
            comps1=image_self.components
            ref_comps=image_data2.components

            if comp_ids=="":
                raise Exception("Please specify valid component IDs with 'comp_ids=...'")
            else:
                if comp_ids=="all":
                    #find all possible component ids
                    comp_ids=[]
                    for comp in image_self.components:
                        comp_ids.append(comp.component_number)
                    for comp in image_data2.components:
                        comp_ids.append(comp.component_number)

                    comp_ids=np.unique(comp_ids)

                comp_ids = [comp_ids] if isinstance(comp_ids,int) else comp_ids
                x_shifts=[]
                y_shifts=[]
                x_shift_err=[]
                y_shift_err=[]
                for comp_id in comp_ids:
                    #get component from comps1:
                    found=False
                    for comp in comps1:
                        if comp.component_number==comp_id:
                            align_comp=comp
                            found=True
                    if not found:
                        align_comp=""
                    found=False
                    for ref_comp in ref_comps:
                        if ref_comp.component_number==comp_id:
                            align_comp_ref=ref_comp
                            found=True
                    if not found:
                        align_comp_ref=""

                    if align_comp!="" and align_comp_ref!="":
                        #this means a component with the given comp_id was found in both images
                        #calculate shift:
                        x1=align_comp.x*image_self.scale
                        x_ref=align_comp_ref.x*image_data2.scale
                        y1=align_comp.y*image_self.scale
                        y_ref=align_comp_ref.y*image_data2.scale

                        x_shifts.append(x1-x_ref)
                        y_shifts.append(y_ref-y1)
                        x_shift_err.append(np.sqrt((align_comp.x_err*image_self.scale)**2+(align_comp_ref.x_err*image_data2.scale)**2))
                        y_shift_err.append(np.sqrt((align_comp.y_err*image_self.scale)**2+(align_comp_ref.y_err*image_data2.scale)**2))
                    else:
                        logger.warning(f"Did no find component with id {comp_id} in both images, skipping it")

                #take mean shift if multiple components were used
                if len(y_shifts)==0:
                    logger.warning("No matching components found, will not apply a shift.")
                    return self
                else:
                    if weight_by_comp_err:
                        # Compute weights as inverse variance
                        weights_x = 1 / np.array(x_shift_err)**2
                        weights_y = 1/ np.array(y_shift_err)**2

                        # Weighted mean
                        x_shift_final = np.sum(weights_x * np.array(x_shifts)) / np.sum(weights_x)
                        y_shift_final = np.sum(weights_y * np.array(y_shifts)) / np.sum(weights_y)
                    else:
                        x_shift_final=np.mean(x_shifts)
                        y_shift_final=np.mean(y_shifts)

                    shift=[y_shift_final/image_self.scale/image_self.degpp,x_shift_final/image_self.scale/image_self.degpp]
                    logger.info('will apply shift (x,y): [{} : {}] {}'.format(-shift[1] * image_self.scale * image_self.degpp,
                                                                       shift[0] * image_self.scale * image_self.degpp,self.unit))


        else:
            warning.warn("Please use valid align method ('cross_correlation','brightest').")


        #shift shifted image
        return image_self.shift(-shift[1]*image_self.scale*image_self.degpp,shift[0]*image_self.scale*image_self.degpp,useDIFMAP=useDIFMAP)

    def restore(self,bmaj=-1,bmin=-1,posa=-1,shift_x=0,shift_y=0,npix="",pixel_size="",weighting=uvw,useDIFMAP=True,mask_outside=False):
        """
        This allows you to restore the ImageData object with a custom beam either with DIFMAP or just the image itself
        Inputs:
            bmaj (float): Beam major axis (in mas)
            bmin (float): Beam minor axis (in mas)
            posa (float): Beam position angle (in deg)
            shift_x (float): Shift in mas in x-direction
            shift_y (float): Shift in mas in y-direction
            npix (int): Number of pixels in one image direction
            pixel_size (float): pixel size in mas
            weighting (list[int]): DIFMAP weighting option
            useDIFMAP (bool): Choose whether to use DIFMAP for the restoring or not
        Returns:
            New ImageData object
        """
        if bmaj==-1:
            bmaj=self.beam_maj
        if bmin==-1:
            bmin=self.beam_min
        if posa==-1:
            posa=self.beam_pa


        #TODO basic sanity check if uvf file is present and if polarization is there
        if self.uvf_file=="" or useDIFMAP==False:
            #this means there is no valid .uvf file or we don't want to use DIFMAP

            logger.warning("No .uvf file attached or useDIFMAP=False selected, will do simple shift of image only")

            # shift in degree
            shift_x_deg = shift_x / self.scale
            shift_y_deg = shift_y / self.scale

            # calculate shift to pixel increments:
            shift_x = -int(shift_x / self.scale / self.degpp)
            shift_y = int(shift_y / self.scale / self.degpp)

            #shift the image mask
            input_ = np.fft.fft2(self.mask)  # before it was np.fft.fftn(img)
            offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
            imgalign = np.fft.ifft2(offset_image)  # again before ifftn
            new_mask = np.real(imgalign) > 0.5

            # shift image directly
            input_ = np.fft.fft2(self.Z)  # before it was np.fft.fftn(img)
            offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
            imgalign = np.fft.ifft2(offset_image)  # again before ifftn
            new_image_i = imgalign.real
            if not (bmaj == -1 and bmin == -1 and posa == -1):
                #convert to jansky per pixel
                new_image_i = JyPerBeam2Jy(new_image_i,self.beam_maj,self.beam_min,self.degpp*self.scale)
                new_image_i = convolve_with_elliptical_gaussian(new_image_i, bmaj / self.scale / self.degpp/(2*np.sqrt(2*np.log(2))),
                                                             bmin / self.scale / self.degpp/(2*np.sqrt(2*np.log(2))), posa)
                #convert to jansky per (new) beam
                new_image_i = Jy2JyPerBeam(new_image_i,bmaj,bmin,self.degpp*self.scale)
            # try polarization
            try:
                input_ = np.fft.fft2(self.stokes_q)  # before it was np.fft.fftn(img)
                offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
                imgalign = np.fft.ifft2(offset_image)  # again before ifftn
                new_image_q = imgalign.real
                if not (bmaj==-1 and bmin ==-1 and posa==-1):
                    new_image_q = JyPerBeam2Jy(new_image_q, self.beam_maj, self.beam_min, self.degpp * self.scale)
                    new_image_q = convolve_with_elliptical_gaussian(new_image_q,
                                                                    bmaj/self.scale/self.degpp/(2*np.sqrt(2*np.log(2))),
                                                                    bmin/self.scale/self.degpp/(2*np.sqrt(2*np.log(2))),posa)
                    # convert to jansky per (new) beam
                    new_image_q = Jy2JyPerBeam(new_image_q, bmaj, bmin, self.degpp * self.scale)

                input_ = np.fft.fft2(self.stokes_u)  # before it was np.fft.fftn(img)
                offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
                imgalign = np.fft.ifft2(offset_image)  # again before ifftn
                new_image_u = imgalign.real
                if not (bmaj==-1 and bmin ==-1 and posa==-1):
                    new_image_u = JyPerBeam2Jy(new_image_u, self.beam_maj, self.beam_min, self.degpp * self.scale)
                    new_image_u= convolve_with_elliptical_gaussian(new_image_u,bmaj/self.scale/self.degpp/(2*np.sqrt(2*np.log(2))),
                                                                    bmin/self.scale/self.degpp/(2*np.sqrt(2*np.log(2))),posa)
                    # convert to jansky per (new) beam
                    new_image_u = Jy2JyPerBeam(new_image_u, bmaj, bmin, self.degpp * self.scale)

            except:
                new_image_q = ""
                new_image_u = ""
                new_stokes_u_fits = ""
                new_stokes_q_fits = ""

            #write outputs to the fitsfiles
            if self.only_stokes_i:
                # this means DIFMAP style fits image
                with fits.open(self.fits_file) as f:
                    f[0].data[0, 0, :, :] = new_image_i
                    new_stokes_i_fits = self.model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                    try:
                        f[1].header['XTENSION'] = 'BINTABLE'
                        #shift model/clean components
                        f[1].data["DELTAX"] += shift_x_deg
                        f[1].data["DELTAY"] += shift_y_deg
                    except:
                        pass
                    if not (bmaj == -1 and bmin == -1 and posa == -1):
                        #Overwrite beam parameters in header
                        f[0].header["BMAJ"] = bmaj / self.scale
                        f[0].header["BMIN"] = bmin / self.scale
                        f[0].header["BPA"] = posa
                    f.writeto(new_stokes_i_fits, overwrite=True)

                if len(self.stokes_q) > 0:
                    with fits.open(self.stokes_q_path) as f:
                        f[0].data[0, 0, :, :] = new_image_q
                        new_stokes_q_fits = self.model_save_dir+"mod_files_q/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'
                            # shift model/clean components
                            f[1].data["DELTAX"] += shift_x_deg
                            f[1].data["DELTAY"] += shift_y_deg
                        except:
                            pass
                        if not (bmaj == -1 and bmin == -1 and posa == -1):
                            # Overwrite beam parameters in header
                            f[0].header["BMAJ"] = bmaj / self.scale
                            f[0].header["BMIN"] = bmin / self.scale
                            f[0].header["BPA"] = posa
                        f.writeto(new_stokes_q_fits, overwrite=True)

                if len(self.stokes_u) > 0:
                    with fits.open(self.stokes_u_path) as f:
                        f[0].data[0, 0, :, :] = new_image_u
                        new_stokes_u_fits = self.model_save_dir+"mod_files_u/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'
                            # shift model/clean components
                            f[1].data["DELTAX"] += shift_x_deg
                            f[1].data["DELTAY"] += shift_y_deg
                        except:
                            pass
                        if not (bmaj == -1 and bmin == -1 and posa == -1):
                            # Overwrite beam parameters in header
                            f[0].header["BMAJ"] = bmaj / self.scale
                            f[0].header["BMIN"] = bmin / self.scale
                            f[0].header["BPA"] = posa
                        f.writeto(new_stokes_u_fits, overwrite=True)


            else:
                # CASA style
                f = fits.open(self.fits_file)
                f[0].data[0, 0, :, :] = new_image_i
                f[0].data[1, 0, :, :] = new_image_q
                f[0].data[2, 0, :, :] = new_image_u
                if not (bmaj == -1 and bmin == -1 and posa == -1):
                    # Overwrite beam parameters in header
                    f[0].header["BMAJ"] = bmaj / self.scale
                    f[0].header["BMIN"] = bmin / self.scale
                    f[0].header["BPA"] = posa
                new_stokes_i_fits = self.model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.fits"
                f.writeto(new_stokes_i_fits, overwrite=True, output_verify='ignore')
                f.close()

                new_stokes_q_fits=""
                new_stokes_u_fits=""

            # if model loaded try shifting model image as well
            try:
                if not self.model_file_path == self.fits_file:
                    input_ = np.fft.fft2(
                        fits.open(self.model_file_path)[0].data[0, 0, :, :])  # before it was np.fft.fftn(img)
                    offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
                    imgalign = np.fft.ifft2(offset_image)  # again before ifftn
                    new_image_model = imgalign.real
                    if not (bmaj == -1 and bmin == -1 and posa == -1):
                        new_image_model = JyPerBeam2Jy(new_image_model, self.beam_maj, self.beam_min,
                                                       self.degpp * self.scale)
                        new_image_model = convolve_with_elliptical_gaussian(new_image_model,
                                                                            bmaj / self.scale / self.degpp / (2*np.sqrt(2*np.log(2))),
                                                                            bmin / self.scale / self.degpp / (2*np.sqrt(2*np.log(2))),
                                                                            posa)
                        # convert to jansky per (new) beam
                        new_image_model = Jy2JyPerBeam(new_image_model, bmaj, bmin, self.degpp * self.scale)

                    with fits.open(self.model_file_path) as f:
                        f[0].data[0, 0, :, :] = new_image_model
                        new_model_fits = self.model_save_dir + "mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                    self.freq / 1e9).replace(".", "_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'
                            f[1].data["DELTAX"] += shift_x_deg
                            f[1].data["DELTAY"] += shift_y_deg
                        except:
                            pass
                        if not (bmaj == -1 and bmin == -1 and posa == -1):
                            f[0].header["BMAJ"] = bmaj / self.scale
                            f[0].header["BMIN"] = bmin / self.scale
                            f[0].header["BPA"] = posa
                        f.writeto(new_model_fits, overwrite=True)
                else:
                    new_model_fits = new_stokes_i_fits
            except:
                new_image_model = ""
                new_model_fits = ""

            new_uvf_file=self.uvf_file

        else:
            #This means we have a valid .uvf file and we will use DIFMAP for shifting and restoring
            # calculate shift to pixel increments:
            shift_x_pix = -int(shift_x / self.scale / self.degpp)
            shift_y_pix = int(shift_y / self.scale / self.degpp)

            #first let's shift the mask
            # shift the image mask
            input_ = np.fft.fft2(self.mask)  # before it was np.fft.fftn(img)
            offset_image = fourier_shift(input_, shift=[shift_y_pix, shift_x_pix])
            imgalign = np.fft.ifft2(offset_image)  # again before ifftn
            new_mask = np.real(imgalign) > 0.5

            #restore Stokes I
            new_stokes_i_fits=self.stokes_i_mod_file.replace(".mod","")

            fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa,shift_x=shift_x,shift_y=shift_y,
                    channel="i",output_dir=self.model_save_dir+"mod_files_clean",outname=new_stokes_i_fits,
                    n_pixel=len(self.X)*2,pixel_size=self.degpp*self.scale,
                    mod_files=[self.stokes_i_mod_file],clean_mod_files=[self.stokes_i_mod_file],
                    uvf_files=[self.uvf_file],weighting=weighting)

            new_stokes_i_fits+=".fits"

            #try to restore modelfit if it is there
            try:
                if not self.model_file_path==self.fits_file:
                    new_model_fits=self.model_mod_file.replace(".mod","")

                    fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                        bmaj=bmaj, bmin=bmin, posa=posa, shift_x=shift_x, shift_y=shift_y,
                        channel="i", output_dir=self.model_save_dir + "mod_files_model", outname=new_model_fits,
                        n_pixel=len(self.X)*2,pixel_size=self.degpp*self.scale,
                        mod_files=[self.model_mod_file], clean_mod_files=[self.stokes_i_mod_file], uvf_files=[self.uvf_file], weighting=weighting)

                    new_model_fits+=".fits"
                else:
                    new_model_fits=new_stokes_i_fits
            except:
                new_model_fits=""

            #try to restore polarization as well if it is there
            try:
                new_stokes_q_fits=self.stokes_q_mod_file.replace(".mod","")
                new_stokes_u_fits=self.stokes_u_mod_file.replace(".mod","")


                fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa,shift_x=shift_x,shift_y=shift_y,
                    channel="q",output_dir=self.model_save_dir+"mod_files_q",outname=new_stokes_q_fits,
                    n_pixel=len(self.X)*2,pixel_size=self.degpp*self.scale,
                    mod_files=[self.stokes_q_mod_file],clean_mod_files=[self.stokes_i_mod_file],
                               uvf_files=[self.uvf_file],weighting=weighting)

                new_stokes_q_fits+=".fits"

                fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa, shift_x=shift_x,shift_y=shift_y,
                    channel="u",output_dir=self.model_save_dir+"mod_files_u",outname=new_stokes_u_fits,
                    n_pixel=len(self.X)*2,pixel_size=self.degpp*self.scale,
                    mod_files=[self.stokes_u_mod_file],clean_mod_files=[self.stokes_i_mod_file],
                               uvf_files=[self.uvf_file],weighting=weighting)

                new_stokes_u_fits+=".fits"

            except:
                new_stokes_q_fits=""
                new_stokes_u_fits=""

            new_uvf_file=new_stokes_i_fits.replace(".fits",".uvf")

        if not self.model_inp:
            new_model_fits = ""

        return ImageData(fits_file=new_stokes_i_fits,
                         uvf_file=new_uvf_file,
                         stokes_q=new_stokes_q_fits,
                         stokes_u=new_stokes_u_fits,
                         mask=new_mask,
                         ridgeline=self.ridgeline,
                         redshift=self.redshift,
                         counter_ridgeline=self.counter_ridgeline,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=new_model_fits,
                         correct_rician_bias=self.correct_rician_bias,
                         comp_ids=self.get_model_info()[0],
                         core_comp_id=self.get_model_info()[1],
                         difmap_path=self.difmap_path,
                         fit_comp_polarization=self.fit_comp_pol)

    def shift(self,shift_x,shift_y,weighting=uvw,useDIFMAP=True):
        """
        Function to shift the image in RA and Dec.

        Args:
            shift_x (float): Shift in Right Ascension (in mas)
            shift_y (float): Shift in Declination (in mas)
            npix (int): Option to change the number of pixels in ONE direction.
            pixel_size (float): Option to change the pixel size (in mas)
            weighting (list[int]): DIFMAP weighting option
            useDIFMAP (bool): Choose whether to use DIFMAP for shifting or not.
        Returns:
            shifted ImageData object
        """
        try:
            #We can just call the restore() function without doing the restore steps
            return self.restore(-1,-1,-1,shift_x,shift_y,weighting=weighting,useDIFMAP=useDIFMAP)
        except:
            raise Exception("No shift possible, something went wrong!")

    def masking(self, mask_type='ellipse', args=False):
        '''
        Function to mask ImageData object.

        Args:
            mask_type: 'npix_x','cut_left','cut_right','radius','ellipse','flux_cut'
            args: the arguments for the mask
                'npix_x': args=[npix_x,npixy]
                'cut_left': args = cut_left
                'cut_right': args = cut_right
                'radius': args = radius
                'ellipse': args = {'e_args': [e_maj,e_min,e_pa], 'e_xoffset': xoff, 'e_yoffset': yoff} all in the image intrinsic unit
                'flux_cut: args = flux cut
        '''

        # cut out inner, optically thick part of the image
        if mask_type == 'npix_x':
            npix_x = args[0]
            npix_y = args[1]
            px_min_x = int(len(self.X) / 2 - npix_x/2)
            px_max_x = int(len(self.X) / 2 + npix_x/2)
            px_min_y = int(len(self.Y) / 2 - npix_y/2)
            px_max_y = int(len(self.Y) / 2 + npix_y/2)

            px_range_x = np.arange(px_min_x, px_max_x + 1, 1)
            px_range_y = np.arange(px_min_y, px_max_y + 1, 1)

            index = np.meshgrid(px_range_y, px_range_x)
            self.mask[tuple(index)] = True

        if mask_type == 'cut_left':
            cut_left = args
            px_max = int(len(self.X) / 2. + cut_left)
            px_range_x = np.arange(0, px_max, 1)
            self.mask[:, px_range_x] = True

        if mask_type == 'cut_right':
            cut_right = args
            px_max = int(len(self.X) / 2 - cut_right)
            px_range_x = np.arange(px_max, len(self.X), 1)
            self.mask[:, px_range_x] = True

        if mask_type == 'radius':
            radius = args
            rr, cc = disk((int(len(self.X) / 2), int(len(self.Y) / 2)), radius)
            self.mask[rr, cc] = True

        if mask_type == 'ellipse':
            e_maj = int(args['e_args'][0]/self.scale/self.degpp)/2
            e_min = int(args['e_args'][1]/self.scale/self.degpp)/2
            e_pa = args['e_args'][2]
            e_xoffset = -int(args['e_xoffset']/self.scale/self.degpp)
            e_yoffset = int(args['e_yoffset']/self.scale/self.degpp)
            try:
                x, y = int(len(self.X) / 2) + e_xoffset, int(len(self.Y) / 2) +e_yoffset
            except:
                try:
                    x, y = int(len(self.X) / 2)+e_xoffset, int(len(self.Y) / 2)
                except:
                    try:
                        x, y = int(len(self.X) / 2) , int(len(self.Y) / 2) + e_yoffset
                    except:
                        x, y = int(len(self.X) / 2) , int(len(self.Y) / 2)

            if e_pa == False:
                e_pa = 0
            else:
                e_pa = e_pa
            rr, cc = ellipse(y, x, e_maj, e_min, rotation=-e_pa * np.pi / 180)
            self.mask[rr, cc] = True

        if mask_type == 'flux_cut':
            flux_cut = args
            self.mask[self.Z>flux_cut*np.max(self.Z)] = True

        if mask_type == 'reset':
            self.mask=np.zeros_like(self.Z)

    def rotate(self,angle,useDIFMAP=True,reshape=False,order=1):
        """
        Function to rotate ImageData Object (note: EVPAs are currently not rotated!)

        Args:
            angle (float): Rotation angle in degrees (North through East)
            useDIFMAP (bool): Choose whether to use DIFMAP or not
            reshape (bool): If useDIFMAP=False, choose whether to reshape the image size to avoid empty areas.
            order (int): Order parameter for scipy.ndimage.rotate function
        Returns:
            rotated ImageData object
        """

        #rotate mask
        new_mask=scipy.ndimage.rotate(self.mask,-angle,reshape=reshape,order=0)
        #make sure values are valid
        new_mask[new_mask < 0.1] = False
        new_mask[new_mask >= 0.1] = True

        #rotate uvf file
        if self.uvf_file!="":
            new_uvf = self.model_save_dir + "mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                self.freq / 1e9).replace(".", "_") + "GHz.uvf"

            rotate_uvf_file(self.uvf_file, -angle, new_uvf)

        #rotate ridgeline
        x_new,y_new=rotate_points(np.array(self.ridgeline.X_ridg), np.array(self.ridgeline.Y_ridg), -angle)
        self.ridgeline.X_ridg=x_new
        self.ridgeline.Y_ridg=y_new

        #rotate counterridgeline
        x_new, y_new = rotate_points(np.array(self.counter_ridgeline.X_ridg), np.array(self.counter_ridgeline.Y_ridg), -angle)
        self.counter_ridgeline.X_ridg = x_new
        self.counter_ridgeline.Y_ridg = y_new

        #do actual image rotations
        if self.uvf_file=="" or not useDIFMAP:
            logger.warning("No .uvf file attached or useDIFMAP=False selected, will do simple shift of image only")

            new_image_i=scipy.ndimage.rotate(self.Z,-angle,reshape=reshape,order=order)

            try:
                new_image_q = scipy.ndimage.rotate(self.stokes_q,-angle,reshape=reshape,order=order)
                new_image_u = scipy.ndimage.rotate(self.stokes_u,-angle,reshape=reshape,order=order)
            except:
                logger.warning("Unable to rotate polarization, probably no polarization loaded")

            # write outputs to the fits files
            if self.only_stokes_i:
                # this means DIFMAP style fits image
                with fits.open(self.fits_file) as f:
                    # overwrite image data
                    f[0].data[0, 0, :, :] = new_image_i
                    new_stokes_i_fits = self.model_save_dir + "mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                        self.freq / 1e9).replace(".", "_") + "GHz.fits"
                    try:
                        f[1].header['XTENSION'] = 'BINTABLE'
                        new_x,new_y=rotate_points(f[1].data["DELTAX"],f[1].data["DELTAY"],-angle)
                        f[1].data['DELTAX']=new_x
                        f[1].data['DELTAY']=new_y
                    except:
                        pass
                    f[0].header['BPA']+=angle
                    f.writeto(new_stokes_i_fits, overwrite=True)

                if len(self.stokes_q) > 0:
                    with fits.open(self.stokes_q_path) as f:
                        # overwrite image data
                        f[0].data[0, 0, :, :] = new_image_q
                        new_stokes_q_fits = self.model_save_dir + "mod_files_q/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                            self.freq / 1e9).replace(".", "_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'
                            new_x, new_y = rotate_points(f[1].data["DELTAX"], f[1].data["DELTAY"], -angle)
                            f[1].data['DELTAX'] = new_x
                            f[1].data['DELTAY'] = new_y
                        except:
                            pass
                        f[0].header['BPA'] += angle
                        f.writeto(new_stokes_q_fits, overwrite=True)
                else:
                    new_stokes_q_fits = ""

                if len(self.stokes_u) > 0:
                    with fits.open(self.stokes_u_path) as f:
                        # overwrite image data
                        f[0].data[0, 0, :, :] = new_image_u
                        new_stokes_u_fits = self.model_save_dir + "mod_files_u/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                            self.freq / 1e9).replace(".", "_") + "GHz.fits"
                        try:
                            f[1].header['XTENSION'] = 'BINTABLE'
                            new_x, new_y = rotate_points(f[1].data["DELTAX"], f[1].data["DELTAY"], -angle)
                            f[1].data['DELTAX'] = new_x
                            f[1].data['DELTAY'] = new_y
                        except:
                            pass
                        f[0].header['BPA'] += angle
                        f.writeto(new_stokes_u_fits, overwrite=True)
                else:
                    new_stokes_u_fits = ""

            else:
                # CASA style
                f = fits.open(self.fits_file)
                # overwrite image data
                f[0].data[0, 0, :, :] = new_image_i
                f[0].data[1, 0, :, :] = new_image_q
                f[0].data[2, 0, :, :] = new_image_u
                new_stokes_i_fits = self.model_save_dir + "mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                    self.freq / 1e9).replace(".", "_") + "GHz.fits"
                f[0].header['BPA'] += angle
                f.writeto(new_stokes_i_fits, overwrite=True, output_verify='ignore')
                new_stokes_q_fits = ""
                new_stokes_u_fits = ""

            # if model loaded try rotating as well
            try:
                if not self.model_file_path == self.fits_file:
                    if not self.model_file_path == "":

                        new_image_model=scipy.ndimage.rotate(fits.open(self.model_file_path)[0].data,-angle,reshape=reshape,order=order)

                        with fits.open(self.model_file_path) as f:
                            f[0].data[0, 0, :, :] = new_image_model
                            new_model_fits = self.model_save_dir + "mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(
                    self.freq / 1e9).replace(".", "_") + "GHz.fits"
                            try:
                                f[1].header['XTENSION'] = 'BINTABLE'
                                new_x, new_y = rotate_points(f[1].data["DELTAX"], f[1].data["DELTAY"], -angle)
                                f[1].data['DELTAX'] = new_x
                                f[1].data['DELTAY'] = new_y
                            except:
                                pass
                            f[0].header['BPA'] += angle
                            f.writeto(new_model_fits, overwrite=True)
                    else:
                        new_model_fits = ""
                else:
                    new_model_fits = new_stokes_i_fits
            except:
                logger.warning("Model not regridded, probably no model loaded.")
                new_model_fits = ""

            if not self.model_inp:
                new_model_fits=""

            self.beam_pa+=angle

            newImageData= ImageData(fits_file=new_stokes_i_fits,
                         uvf_file=self.uvf_file,
                         stokes_q=new_stokes_q_fits,
                         stokes_u=new_stokes_u_fits,
                         mask=new_mask,
                         redshift=self.redshift,
                         ridgeline=self.ridgeline,
                         counter_ridgeline=self.counter_ridgeline,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=new_model_fits,
                         correct_rician_bias=self.correct_rician_bias,
                         comp_ids=self.get_model_info()[0],
                         core_comp_id=self.get_model_info()[1],
                         difmap_path=self.difmap_path,
                         fit_comp_polarization=self.fit_comp_pol)

        else:

            if not self.model_inp:
                self.model_file_path=""

            newImageData=ImageData(fits_file=self.fits_file,
                         uvf_file=self.uvf_file,
                         stokes_q=self.stokes_q_path,
                         stokes_u=self.stokes_u_path,
                         mask=self.mask,
                         redshift=self.redshift,
                         ridgeline=self.ridgeline,
                         counter_ridgeline=self.counter_ridgeline,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=self.model_file_path,
                         correct_rician_bias=self.correct_rician_bias,
                         comp_ids=self.get_model_info()[0],
                         core_comp_id=self.get_model_info()[1],
                         difmap_path=self.difmap_path,
                         fit_comp_polarization=self.fit_comp_pol)

            rotate_mod_file(self.stokes_i_mod_file,angle,self.stokes_i_mod_file)
            try:
                rotate_mod_file(self.stokes_q_mod_file,angle,self.stokes_q_mod_file)
                rotate_mod_file(self.stokes_u_mod_file,angle,self.stokes_u_mod_file)
            except:
                logger.debug("Could not rotate polarization, probably not loaded.")
            try:
                rotate_mod_file(self.model_mod_file,angle,self.model_mod_file)
            except:
                logger.debug("Could not rotate model, probably not loaded.")

            newImageData.uvf_file=new_uvf
            newImageData.mask=new_mask
            newImageData.beam_pa+=angle

            newImageData=newImageData.restore()

        return newImageData

    def get_peak_distance(self):
        """
        Function to calculate the Distance between Stokes I and Linear Polarization Peak
        Returns:
            [x_dist,y_dist]: Vector difference between Stokes I and Lin-Pol peak (in mas)
        """
        #returns distance between stokes I and lin pol peak

        #find maximum indices for stokes I and lin_pol
        y_i, x_i = np.unravel_index(np.argmax(self.Z), self.Z.shape)
        y_pol, x_pol = np.unravel_index(np.argmax(self.lin_pol),self.lin_pol.shape)

        x_dist=self.X[x_pol]-self.X[x_i]
        y_dist=self.Y[y_pol]-self.Y[y_i]

        return [x_dist, y_dist]

    def center(self,mode="stokes_i",useDIFMAP=True):
        """
        Function to center the brightest pixel of the image.

        Args:
            mode: Choose which map to use ('stokes_i', 'lin_pol')
            useDIFMAP: Choose whether to use DIFMAP or not.
        Returns:
            Shifted ImageData object
        """

        if mode=="stokes_i" or mode=="lin_pol" or mode=="linpol":
            if mode=="stokes_i":
                ref_image=self.Z
            elif mode=="lin_pol" or mode=="linpol":
                ref_image=self.lin_pol

            # find brightest pixel of reference image and center of current image
            x_ind, y_ind = int(len(self.X)/2),int(len(self.Y)/2)
            x_, y_ = np.unravel_index(np.argmax(ref_image), ref_image.shape)

            shift = [y_ind - y_, x_ind - x_]
            logger.info('will apply shift (x,y): [{} : {}] {}'.format(-shift[1] * self.scale * self.degpp,
                                                                 shift[0] * self.scale * self.degpp,self.unit))

            return self.shift(-shift[1] * self.scale * self.degpp,
                              shift[0] * self.scale * self.degpp, useDIFMAP=useDIFMAP)
        elif mode == "core":
            core = self.get_core_component()
            return self.shift(-core.x*core.scale,-core.y*core.scale,useDIFMAP=useDIFMAP)
        else:
            raise Exception("Please pick valid 'mode' parameter ('stokes_i','lin_pol','core').")

    def get_profile(self,point1,point2,show=True,image="stokes_i"):
        """
        Function to obtain a line profile of the image.

        Args:
            point1 (list[float]): Starting Point of the profile [x1,y1] (in mas)
            point2 (list[float]): End Point of the profile [x2,y2] (in mas)
            show (bool): Choose whether to display a plot of the profile
            image (bool): Choose map to use ('stokes_i','lin_pol','evpa','spix','rm')
        Returns:
            x_values, intensity_profile: Array of the Distance from point1 to point2 and the profile
        """

        #get index of slice ends
        x_ind1 = closest_index(self.X,point1[0])
        y_ind1 = closest_index(self.Y,point1[1])
        x_ind2 = closest_index(self.X, point2[0])
        y_ind2 = closest_index(self.Y, point2[1])

        #select image to get slice from
        if image=="stokes_i":
            image_data=self.Z
        elif image=="lin_pol":
            image_data=self.lin_pol
        elif image=="evpa":
            image_data=self.evpa
        elif image=="spix":
            image_data=self.spix
        elif image=="rm":
            image_data=self.rm
        elif image=="frac_pol":
            image_data=self.lin_pol/self.Z
        else:
            raise Exception(f"Please specify valid 'image' parameter, image='{image}' not supported.")

        intensity_profile=profile_line(image_data, (y_ind1,x_ind1), (y_ind2,x_ind2))

        #calculate distance between points
        dist=np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        #get x_values of intensity_profile
        x_values=np.linspace(0,dist,len(intensity_profile))

        if show:
            plt.plot(x_values,intensity_profile)
            plt.xlabel(f"Distance from Point 1 [{self.unit}]")
            plt.ylabel("Flux Density [Jy/beam]")
            plt.tight_layout()
            plt.show()

        return x_values, intensity_profile

    def get_ridgeline(self,method="slices",angle_for_slices=0,auto_rotate=True,jet_angle="",
                      cut_radial=5.0, cut_final=10.0,counterjet=True,width=40,j_len="",start_radius=0,chi_sq_val=100.0,err_FWHM=0.1):

        """
        Function to calculate the Ridgeline (and Counter-Ridgeline) of an image.

        Args:
            method (str): Select method to use ('slices', 'polar')
            angle_for_slices (float): Choose angle for the slices method
            auto_rotate (bool): For the 'slices' method, choose whether to automatically detect the jet direction
            jet_angle (float): If auto_rotate=False, provide the jet_angle in degrees for the 'slices' method
            cut_radial (float): radial SNR Cut for the 'slices' method
            cut_final (float): final SNR cut for the 'slices' method
            counterjet (bool): Choose whether to also fit a counterjet
            width (int): Jet width in to consider for 'slices' method (in pixel)
            j_len (int): Jet length to consider for 'slices' method (in pixel)
            start_radius (float): Start radius for polar method (in mas)
            chi_sq_val (float): Chi-squared cut for fits.
            err_FWHM (float): Relative error of the FWHM to consider for fits
        Returns:
            Ridgeline and Counter-Ridgeline objects
        """

        if method=="slices":
            #this is Lucas method with an additional option to auto_rotate.
            image=self.copy()

            if auto_rotate:
                #convert image to polar coordinates
                R, Theta, Z_polar = convert_image_to_polar(self.X, self.Y, self.Z)
                #Integrate over the radius to find jet direction:
                integrated_jet=np.zeros(len(Theta[:,0]))
                for i in range(len(R[0])):
                    integrated_jet+=Z_polar[:,i]*R[:,i] #correct for rdTheta in integration
                #plt.plot(Theta[:,0],integrated_jet)
                #plt.show()
                #find maximum flux
                max_ind=np.argmax(integrated_jet)
                jet_direction=Theta[:,0][max_ind]
                logger.info(f"Automatically determined jet direction {jet_direction}°.")
                image=image.rotate(-jet_direction)
            elif jet_angle!="":
                image=image.rotate(-jet_angle)
            else:
                logger.warning("Will assume the jet was already rotated to position angle 0°.")

            # TODO need to CONVERT IT TO Jy/px????
            image_data = image.Z

            #if not j_len given, will use full image - 10 pixels at the edge
            if j_len=="":
                j_len=int(len(self.Y)/2-10)

            #get ridgeline
            ridgeline=Ridgeline().get_ridgeline_luca(image_data,self.noise,self.error,self.degpp*self.scale,[self.beam_maj,self.beam_min,self.beam_pa],
                                                     self.X,self.Y,angle_for_slices=angle_for_slices,cut_radial=cut_radial,
                                                     cut_final=cut_final,width=width,j_len=j_len,chi_sq_val=chi_sq_val,err_FWHM=err_FWHM)
            image.ridgeline=ridgeline

            if counterjet:
                counter_ridgeline=Ridgeline().get_ridgeline_luca(image_data,self.noise,self.error,self.degpp*self.scale,[self.beam_maj,self.beam_min,self.beam_pa],
                                                     self.X,self.Y,counterjet=True,angle_for_slices=angle_for_slices,cut_radial=cut_radial,
                                                     cut_final=cut_final,width=width,j_len=j_len,chi_sq_val=chi_sq_val,err_FWHM=err_FWHM)

                image.counter_ridgeline=counter_ridgeline

            if auto_rotate:
                # rotate image back
                image.rotate(jet_direction)
            elif jet_angle!="":
                image = image.rotate(jet_angle)
            # set new ridgeline
            self.ridgeline = image.ridgeline
            self.counter_ridgeline = image.counter_ridgeline

            return self.ridgeline, self.counter_ridgeline

        elif method=="polar":
            #convert image to polar coordinates
            image = self.copy()

            if auto_rotate:
                # convert image to polar coordinates
                R, Theta, Z_polar = convert_image_to_polar(self.X, self.Y, self.Z)
                # Integrate over the radius to find jet direction:
                integrated_jet = np.zeros(len(Theta[:, 0]))
                for i in range(len(R[0])):
                    integrated_jet += Z_polar[:, i] * R[:, i]  # correct for rdTheta in integration
                # plt.plot(Theta[:,0],integrated_jet)
                # plt.show()
                # find maximum flux
                max_ind = np.argmax(integrated_jet)
                jet_direction = Theta[:, 0][max_ind]
                logger.info(f"Automatically determined jet direction {jet_direction}°.")
                image = image.rotate(-jet_direction)
            elif jet_angle != "":
                image = image.rotate(-jet_angle)
            else:
                logger.warning("Will assume the jet was already rotated to position angle 0°.")

            R, Theta, Z_polar = convert_image_to_polar(image.X, image.Y, image.Z)

            ridgeline=Ridgeline().get_ridgeline_polar(R,Theta,Z_polar,self,[self.beam_maj,self.beam_min,self.beam_pa],self.error,
                                                      start_radius=start_radius)

            image.ridgeline=ridgeline

            if auto_rotate:
                # rotate image back
                image.rotate(jet_direction)
            elif jet_angle != "":
                image = image.rotate(jet_angle)
            # set new ridgeline
            self.ridgeline = image.ridgeline

            return self.ridgeline, self.counter_ridgeline

        elif method=="polar_gauss":
            #convert image to polar coordinates
            R, Theta, Z_polar = convert_image_to_polar(self.X, self.Y, self.Z)

            ridgeline=Ridgeline().get_ridgeline_polar(R,Theta,Z_polar,[self.beam_maj,self.beam_min,self.beam_pa],self.error,
                                                      start_radius=start_radius)

            self.ridgeline=ridgeline

            return self.ridgeline, self.counter_ridgeline
        else:
            raise Exception("Please select valid ridgeline method ('polar', 'slices').")

    def get_noise_from_shift(self,shift_factor=20):
        """
        Function to calculate the image noise by shifting the phase center with DIFMAP

        Args:
            shift_factor (float): Factor of how far times the image size to shift the phase center away.
        Returns:
            Noise value in Jy
        """

        if self.uvf_file == "":
            logger.warning("Shift not possible, no .uvf file attached to ImageData!")
            return self.noise

        size_x=len(self.X)*self.degpp*self.scale
        size_y=len(self.Y)*self.degpp*self.scale

        #shift data away to get rms
        shifted_image=self.shift(size_x*shift_factor,size_y*shift_factor)

        noise=np.std(shifted_image.Z)

        return noise

    def jet_to_counterjet_profile(self,savefig="",show=True):
        """
        Function to plot the jet-to-counterjet ratio

        Args:
            savefig (str): File path to store the plot
            show (bool): Choose whether to display the plot
        """
        self.ridgeline.jet_to_counterjet_profile(self.counter_ridgeline,savefig=savefig,show=show)

    def get_model_info(self):
        """
        Helper method to get the current state of the model
        Returns:
            List of Component IDs and the Core Component ID
        """
        comp_ids=[]
        core_comp_id=0
        if self.components!=[]:
            for comp in self.components:
                comp_ids.append(comp.component_number)
                if comp.is_core:
                    core_comp_id=comp.component_number

        return comp_ids, core_comp_id

    def change_component_ids(self,old_ids,new_ids):
        """
        Function to assign new component numbers

        Args:
            old_ids (int or list[int]): Old component IDs
            new_ids (int or list[int]): New component IDs
        """

        #handle single value input
        if isinstance(old_ids,int) and isinstance(new_ids,int):
            old_ids=[old_ids]
            new_ids=[new_ids]

        old_ids=np.array(old_ids)
        new_ids=np.array(new_ids)

        if len(np.unique(old_ids)) != len(old_ids) or len(np.unique(new_ids)) != len(new_ids):
            raise Exception("Component number specified more than one time in old_ids or new_ids!")

        #set new component ids
        for ind,comp in enumerate(self.components):
            if comp.component_number in old_ids:
                i=int(np.where(np.array(old_ids)==comp.component_number)[0][0])
                self.components[ind].component_number=new_ids[i]
            else:
                if comp.component_number in new_ids:
                    #in that case we will reset the component id to avoid duplication
                    self.components[ind].component_number=-1

    def set_core_component(self,id):
        """
        Function to set the core component

        Args:
            id (int): Component ID of the core component
        """

        core_ind=""
        for ind, comp in enumerate(self.components):
            if comp.component_number==id:
                self.components[ind].is_core=True
                core_ind=ind
            else:
                self.components[ind].is_core=False

        if core_ind=="":
            logger.warning(f"No component with ID {id} found, no core currently set!")
        else:
            #recalculate core distances for every component
            for i, comp in enumerate(self.components):
                core=self.components[core_ind]
                self.components[i].set_distance_to_core(core.x, core.y,core.x_err,core.y_err)

    def get_component(self,id):
        """
        Function to get a specific Component.

        Args:
            id (int): ID of the component
        Returns:
            Component
        """
        for comp in self.components:
            if comp.component_number==id:
                return comp

        raise Exception(f"Component with ID {id} not found.")

    def get_core_component(self):
        """
        Function to retrieve the core component.
        Returns:
            Core Component
        """
        for comp in self.components:
            if comp.is_core:
                return comp

        raise Exception(f"No core component defined.")

    def remove_component(self,id):
        """
        Function to remove the core component from the Stokes I image
        """

        if isinstance(id,int):
            id=[id]
        elif not isinstance(id,list):
            raise Exception("Please enter valid component id (int or list[int])!")

        comps_to_remove=[]
        for i in id:
            comps_to_remove.append(self.get_component(i))

        #TODO rewrite to work without ehtim
        import ehtim as eh
        mod=eh.model.Model()

        for comp in comps_to_remove:
            mod=mod.add_gauss(F0=comp.flux,
                              FWHM_maj=comp.maj*comp.scale*eh.RADPERUAS*1e3,
                              FWHM_min=comp.min*comp.scale*eh.RADPERUAS*1e3,
                              PA=comp.pos/180*np.pi,
                              x0=comp.x/180*np.pi,
                              y0=comp.y/180*np.pi)

        im=mod.make_image((np.max(self.X)-np.min(self.X))*1e3*eh.RADPERUAS, len(self.X))
        im=im.blur_gauss([self.beam_maj/self.scale/180*np.pi,self.beam_min/self.scale/180*np.pi,self.beam_pa/180*np.pi])

        image=im.imvec.reshape((im.ydim, im.xdim))
        image=Jy2JyPerBeam(image,self.beam_maj,self.beam_min,self.degpp*self.scale)
        image=np.flip(image,axis=0)

        #subtract core from stokes I image
        self.Z=np.array(self.Z)-image

        return self

    def calculate_opening_angle(self,ids="", snr_cut=1):
        """
        Calculates the opening angle for circular Gauss components between the core component and a given component
        Args:
            ids (int, list[int]): Component ID of component to calculate the opening angle for

        Returns:
            angle (list[float]): Opening angles in degrees
        """

        if isinstance(ids,list):
            ids=ids
        elif isinstance(ids,int):
            ids=[ids]
        else:
            if not isinstance(ids,str) or ids!="":
                raise Exception("Invalid IDs provided.")
            else:
                ids,core_id=self.get_model_info()
                ids.remove(core_id)

        core=self.get_core_component()
        angles = []

        for id in ids:
            if id in self.get_model_info()[0]:
                comp = self.get_component(id)

                if isinstance(comp,Component) and comp.resolved and comp.snr>=snr_cut:

                    comp_dist=comp.maj*comp.scale/2
                    if core.resolved:
                        core_dist=core.maj*comp.scale/2
                    else:
                        core_dist=core.res_lim_maj*comp.scale/2
                    delta_x = (comp.x - core.x) * comp.scale
                    delta_y = (comp.y - core.y) * comp.scale

                    """
                    #this part allows to also do this calculation with elliptical components, but we should discuss if we want it like this
                    def calculate_theta():
                        if (delta_y > 0 and delta_x > 0) or (delta_y > 0 and delta_x < 0):
                            return np.arctan(delta_x / delta_y) / np.pi * 180
                        elif delta_y < 0 and delta_x > 0:
                            return np.arctan(delta_x / delta_y) / np.pi * 180 + 180
                        elif delta_y < 0 and delta_x < 0:
                            return np.arctan(delta_x / delta_y) / np.pi * 180 - 180
                        else:
                            return 0
        
                    theta = calculate_theta()
        
                    # check core resolution limit
                    theta_maj, theta_min = get_resolution_limit(self.beam_maj, self.beam_min, self.beam_pa, theta, core.snr,
                                                                method=res_lim_method, weighting=uvw)
        
                    new_pos=theta-comp.pos+90
                    new_pos_core=theta-core.pos+90
        
                    line_comp = Line(Point(0, 0), Point(np.cos(new_pos / 180 * np.pi), np.sin(new_pos / 180 * np.pi)))
                    line_core = Line(Point(0, 0), Point(np.cos(new_pos_core / 180 * np.pi), np.sin(new_pos_core / 180 * np.pi)))
        
                    core_Ellipse=Ellipse(Point(0,0),hradius=core.maj*comp.scale/2,vradius=core.min*comp.scale/2)
                    comp_Ellipse=Ellipse(Point(0,0),hradius=comp.maj*comp.scale/2,vradius=comp.min*comp.scale/2)
        
                    if core.maj==0 or core.min==0:
                        core_dist=np.abs(theta_maj/2)
                    else:
                        p1, p2 = core_Ellipse.intersect(line_core)
                        core_dist=np.abs(float(p1.distance(p2))/2)
                    p1, p2 = comp_Ellipse.intersect(line_comp)
                    comp_dist=np.abs(float(p1.distance(p2))/2)
                    """

                    dist=np.sqrt(delta_x**2+delta_y**2)
                    #calculate opening angle
                    angle=np.arctan((comp_dist-core_dist)/dist)/np.pi*180*2

                    angles.append(angle)

                else:
                    logger.debug(f"Component {comp.component_number} unresolved, will not calculate opening angle.")
            else:
                logger.debug(f"Component {id} not found, will skip it.")

        return angles

    def fit_comp_polarization(self):
        write_mod_file_from_components(self.components,channel="i",export="tmp/model_q.mod",adv=[True])
        os.system("cp tmp/model_q.mod tmp/model_u.mod")
        comps_q=copy.deepcopy(self.components)
        comps_u=copy.deepcopy(self.components)
        comps_q=modelfit_difmap(self.uvf_file,"tmp/model_q.mod",50,difmap_path,components=comps_q,
                                weighting=uvw,channel="q",do_selfcal=True,selfcal_model=self.stokes_i_mod_file)
        comps_u=modelfit_difmap(self.uvf_file,"tmp/model_u.mod",50,difmap_path,components=comps_u,
                                weighting=uvw,channel="u",do_selfcal=True,selfcal_model=self.stokes_i_mod_file)

        for j,comp in enumerate(self.components):
            for i in range(len(comps_q)):
                #we need to check the component association (just to be sure)
                if abs(comps_q[i].x-comp.x)<1e-4/comp.scale and abs(comps_q[i].y-comp.y)<1e-4/comp.scale and abs(comps_q[i].maj-comp.maj)<1e-4/comp.scale:
                    #calculate lin_pol and EVPA from Q and U flux
                    lin_pol=np.sqrt(comps_q[i].flux**2+comps_u[i].flux**2)
                    evpa=0.5*np.arctan2(comps_u[i].flux,comps_q[i].flux)/np.pi*180
                    #set lin_pol and evpa of component
                    self.components[j].lin_pol = lin_pol
                    self.components[j].evpa = evpa

    def fit_collimation_profile(self,method="model",jet="Jet",fit_type='brokenPowerlaw',x0=False,s=100,
                                plot_data=True,plot_fit=True,fit_r0=True,plot="",show=False,label="",color=plot_colors[0],marker=plot_markers[0]):
        """
        Function to fit a collimation profile to the jet/counterjet

        Args:
            method (str): Method to use for collimation profile ('model' to use model components, 'ridgeline' to use ridgeline fit)
            jet (str): Choose whether to do Jet ('Jet'), Counterjet ('Cjet') or both ('Twin')
            fit_type (str): Choose fit_type to use ('brokenPowerlaw' or 'Powerlaw')
            x0_bpl (list[float]): Start values for fit
            plot_data (bool): Choose whether to plot the fitted data
            plot_fit (bool): Choose whether to plot the fit
            fit_r0 (bool): Choose whether to include (r+r0) in fit or just r
            plot (JetProfilePlot): Pass JetProfilePlot to add plots, default will create a new one
            show (bool): Choose whether to show the plot
            label (str): Label for the fitted data/fit
            color (str): Plot color
            marker (str): Plot marker
        Returns:
            plot (JetProfilePlot)

        """

        fit_fail_jet=False
        fit_fail_counterjet=False

        if method=="model":
            #TODO make it work also for counterjet
            #jet info
            dists=[]
            widths=[]
            width_errs=[]

            #counter jet info
            cdists = []
            cwidths = []
            cwidth_errs = []

            for comp in self.components:
                #if component Jet
                dists.append(comp.distance_to_core*self.scale)
                widths.append(comp.maj*self.scale)
                width_errs.append(comp.maj_err*self.scale)
                #else component counterjet
                    #cdists.append(comp.distance_to_core * self.scale)
                    #cwidths.append(comp.maj * self.scale)
                    #cwidth_errs.append(comp.maj_err * self.scale)

        elif method=="ridgeline":

            #jet info
            dists=self.ridgeline.dist
            widths=self.ridgeline.width
            width_errs=self.ridgeline.width_err

            #counterjet info
            cdists = self.counter_ridgeline.dist
            cwidths = self.counter_ridgeline.width
            cwidth_errs = self.counter_ridgeline.width_err

        else:
            raise Exception("Please specify valid 'method' for fit_collimation_profile ('model', 'ridgeline').")

        if jet=="Jet" or jet=="Twin":
            try:
                beta, sd_beta, chi2, out = fit_width(dists, widths, width_err=width_errs, dist_err=False,s=s,
                                                     fit_type=fit_type,x0=x0,fit_r0=fit_r0)
            except:
                logger.warning("Collimation fit did not work for jet!")
                fit_fail_jet=True

        if jet=="CJet" or jet=="Twin":
            try:
                cbeta, csd_beta, cchi2, cout = fit_width(cdists, cwidths, width_err=cwidth_errs, dist_err=False,s=s,
                                                     fit_type=fit_type,x0=x0,fit_r0=fit_r0)
            except:
                logger.warning("Collimation fit did not work for counter jet!")
                fit_fail_counterjet=True

        if plot=="":
            plot=JetProfilePlot(jet=jet)
        else:
            try:
                if plot.jet != jet:
                    raise Exception("Plot has wrong 'jet' type.")
            except:
                raise Exception("Plot is not a valid 'JetProfilePlot'.")

        if plot_data:
            if jet=="Jet":
                plot.plot_profile(dists,widths,width_errs,color,marker,label=label)
            elif jet=="CJet":
                plot.plot_profile(cdists,cwidths,cwidth_errs,color,marker,label=label)
            else:
                plot.plot_profile([dists,cdists],[widths,cwidths],[width_errs,cwidth_errs],color,marker,label=label)

        x=np.linspace(min(dists),max(dists),1000)
        if plot_fit:
            if jet=="Jet" or jet=="Twin":
                if not fit_fail_jet:
                    plot.plot_fit(x, fit_type, beta, sd_beta, chi2, "Jet", color, label=label,fit_r0=fit_r0,s=s)
            if jet=="CJet" or jet=="Twin":
                if not fit_fail_counterjet:
                    plot.plot_fit(x, fit_type, cbeta, csd_beta, cchi2, "CJet", color, label=label,fit_r0=fit_r0,s=s)


        if show:
            plot.plot_legend()
            plt.show()

        return plot
