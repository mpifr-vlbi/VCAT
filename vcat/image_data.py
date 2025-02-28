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

from vcat.graph_generator import FitsImage
from vcat.kinematics import Component
from vcat.alignment.align_imagesEHTim_final import AlignMaps
from vcat.helpers import *
from vcat.stacking_helpers import fold_with_beam
import warnings
from scipy.ndimage import fourier_shift, shift
from skimage.draw import disk, ellipse
from skimage.registration import phase_cross_correlation
from scipy.interpolate import RegularGridInterpolator

class ImageData(object):

    """ Class to handle VLBI Image data (single image with or without polarization at one frequency)
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
                 mask="",
                 stokes_q="",
                 stokes_u="",
                 model_save_dir="tmp/",
                 is_casa_model=False,
                 noise_method="Histogram Fit", #choose noise method
                 correct_rician_bias=False,
                 difmap_path=""):

        self.file_path = fits_file
        self.fits_file = fits_file
        self.model_file_path = model
        self.lin_pol=lin_pol
        self.evpa=evpa
        self.stokes_i=stokes_i
        self.uvf_file=uvf_file
        self.difmap_path=difmap_path
        self.residual_map_path=""
        self.noise_method=noise_method
        self.is_casa_model=is_casa_model
        self.model_save_dir=model_save_dir
        self.correct_rician_bias=correct_rician_bias


        # Read clean files in
        if fits_file!="":
            hdu_list=fits.open(fits_file)
            self.hdu_list = hdu_list
            self.no_fits=False
        else:
            self.no_fits=True
        
        self.stokes_q_path=stokes_q
        self.stokes_u_path=stokes_u
        stokes_q_path=stokes_q
        stokes_u_path=stokes_u
        #read stokes data from input files if defined
        if stokes_q != "":
            try:
                q_fits=fits.open(stokes_q)
                stokes_q = q_fits[0].data[0, 0, :, :]
                q_fits.close()
            except:
                stokes_q=stokes_q
        else:
            stokes_q=[]

        if stokes_u != "":
            try:
                u_fits=fits.open(stokes_u)
                stokes_u = u_fits[0].data[0, 0, :, :]
                u_fits.close()
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
        self.only_stokes_i = False
        if hdu_list[0].data.shape[0] == 1:
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
        self.mjd = Time(self.date).mjd
        try:
            self.difmap_noise = float(hdu_list[0].header["NOISE"])
        except:
            self.difmap_noise = 0
        

        try:
            q_fits=fits.open(stokes_q_path)
            u_fits=fits.open(stokes_q_path)
            self.difmap_pol_noise = np.sqrt(float(q_fits[0].header["NOISE"])**2+float(u_fits[0].header["NOISE"])**2)
            q_fits.close()
            u_fits.close()
        except:
            self.difmap_pol_noise = 0
    
        #calculate image noise according to the method selected
        unused, levs_i = get_sigma_levs(self.Z, 1,noise_method=self.noise_method,noise=self.difmap_noise) #get noise for stokes i
        try:
            unused, levs_pol = get_sigma_levs(self.lin_pol, 1,noise_method=self.noise_method,noise=self.difmap_noise) #get noise for polarization
        except:
            levs_pol=[0]

        # calculate image noise
        unused, levs_i_3sigma = get_sigma_levs(self.Z, 3,noise_method=self.noise_method,noise=self.difmap_noise)  # get noise for stokes i
        try:
            unused, levs_pol_3sigma = get_sigma_levs(self.lin_pol, 3,noise_method=self.noise_method,noise=self.difmap_noise)  # get noise for polarization
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
            os.makedirs(model_save_dir+"mod_files_model/",exist_ok=True)
            self.model_mod_file=model_save_dir+"mod_files_model/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(self.model, self.model_mod_file, freq=self.freq)
        if is_casa_model:
            #TODO basic checks if file is valid
            os.makedirs(model_save_dir,exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_clean", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_q", exist_ok=True)
            os.makedirs(model_save_dir + "mod_files_u", exist_ok=True)
            self.stokes_i_mod_file=model_save_dir+"mod_files_clean/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file_from_casa(self.file_path,channel="i", export=self.stokes_i_mod_file)    
            self.stokes_q_mod_file=model_save_dir+"mod_files_q/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file_from_casa(self.file_path,channel="q", export=self.stokes_q_mod_file)
            self.stokes_u_mod_file=model_save_dir+"mod_files_u/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file_from_casa(self.file_path,channel="u", export=self.stokes_u_mod_file)
        else:
            self.model=None
        try:
            os.makedirs(model_save_dir+"mod_files_clean", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_q", exist_ok=True)
            os.makedirs(model_save_dir+"mod_files_u", exist_ok=True)
            #try to import model which is attached to the main .fits file
            model_i = getComponentInfo(fits_file)
            if self.model==None:
                self.model = model_i
            self.stokes_i_mod_file=model_save_dir+"mod_files_clean/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(model_i, self.stokes_i_mod_file, freq=self.freq)
            #load stokes q and u clean models
            self.model_q=getComponentInfo(stokes_q_path)
            self.stokes_q_mod_file=model_save_dir+"mod_files_q/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(self.model_q, self.stokes_q_mod_file, freq=self.freq)
            self.model_u=getComponentInfo(stokes_u_path)
            self.stokes_u_mod_file=model_save_dir+"mod_files_u/"+ self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod"
            write_mod_file(self.model_u, self.stokes_u_mod_file, freq=self.freq)
        except:
            pass

        #save modelfit (or clean) components as Component objects
        self.components=[]

        for ind,comp in self.model.iterrows():
            component=Component(comp["Delta_x"],comp["Delta_y"],comp["Major_axis"],comp["Minor_axis"],
                    comp["PA"],comp["Flux"],comp["Date"],comp["mjd"],comp["Year"],freq=self.freq)
            self.components.append(component)
        
        #calculate residual map if uvf and modelfile present
        if self.uvf_file!="" and self.model_file_path!="" and not is_casa_model:
            self.residual_map_path = model_save_dir + "residual_maps/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq / 1e9).replace(".",
                                                                                                                 "_") + "GHz_residual.fits"
            get_residual_map(self.uvf_file,model_save_dir+ "residual_maps/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod",
                             difmap_path=self.difmap_path,
                             save_location=self.residual_map_path,npix=len(self.X),pxsize=self.degpp)

        hdu_list.close()

        #calculate cleaned flux density from mod files
        #first stokes I
        try:
            self.integrated_flux_clean=total_flux_from_mod(self.model_save_dir+"mod_files_clean/"  + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
        except:
            self.integrated_flux_clean = 0
        #and then polarization
        try:
            flux_q=total_flux_from_mod(self.model_save_dir+"mod_files_q/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            flux_u=total_flux_from_mod(self.model_save_dir+"mod_files_u/" + self.name + "_" + self.date + "_" + "{:.0f}".format(self.freq/1e9).replace(".","_") + "GHz.mod")
            self.integrated_pol_flux_clean=np.sqrt(flux_u**2+flux_q**2)
            self.frac_pol = image_data.integrated_pol_flux_clean / image_data.integrated_pol_flux_clean
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
        else:
            if np.shape(mask) != np.shape(self.Z):
                warnings.warn("Mask input format invalid, Mask reset to no mask.",UserWarning)
                self.mask = np.zeros_like(self.Z, dtype=bool)
            else:
                self.mask=mask

    #print function for ImageData
    def __str__(self):
        try:
            freq_ghz="{:.1f}".format(self.freq*1e-9)
            line1= f"Image of the source {self.name} at frequency {freq_ghz} GHz on {self.date} \n"
            line2= f"Total cleaned flux: {self.integrated_flux_clean*1000:.3f} mJy \n"
            line3= f"Image Noise: {self.noise*1000:.3f} mJy using method '{self.noise_method}'\n"
            line4= f"Pol Flux: {self.integrated_pol_flux_clean*1000:.3f} mJy ({self.frac_pol*100:.2f}%)\n"
            line5= f"Average EVPA direction: {self.evpa_average/np.pi*180:.2f}"

            return line1+line2+line3+line4+line5
        except:
            return "No data loaded yet."

    def regrid(self,npix="",pixel_size="",weighting=[0,-1],useDIFMAP=True,mask_outside=False):
        """
        This method regrids the image in full polarization
        Args:
            npix: Number of pixels in ONE direction
            pixel_size: Size of pixel in image scale units (usually mas)
            weighting: DIFMAP style weighting
            useDIFMAP: Choose whether to regrid using DIFMAP or not
            mask_outside: Choose whether new image ares created through regridding will be masked automatically (bool)

        Returns:
            ImageData object with regridded images

        """

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
                warnings.warn("Unable to regrid polarization, probably no polarization loaded", UserWarning)

            # write outputs to the fitsfiles
            if self.only_stokes_i:
                # this means DIFMAP style fits image
                with fits.open(self.fits_file) as f:
                    #overwrite image data
                    f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                    f[0].data[0, 0, :, :] = new_image_i
                    new_stokes_i_fits = self.fits_file.replace(".fits", "_convolved.fits")
                    f[1].header['XTENSION'] = 'BINTABLE'
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
                        new_stokes_q_fits = self.stokes_q_path.replace(".fits", "_convolved.fits")
                        f[1].header['XTENSION'] = 'BINTABLE'
                        # modify header parameters to new npix and pixelsize
                        f[0].header["NAXIS1"] = npix
                        f[0].header["NAXIS2"] = npix
                        f[0].header["CDELT1"] = -pixel_size / self.scale
                        f[0].header["CDELT2"] = pixel_size / self.scale
                        f[0].header["CRPIX1"] = int(f[0].header["CRPIX1"] / len(self.X) * npix)
                        f[0].header["CRPIX2"] = int(f[0].header["CRPIX2"] / len(self.X) * npix)
                        f.writeto(new_stokes_q_fits, overwrite=True)

                if len(self.stokes_u) > 0:
                    with fits.open(self.stokes_u_path) as f:
                        # overwrite image data
                        f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                        f[0].data[0, 0, :, :] = new_image_u
                        new_stokes_u_fits = self.stokes_u_path.replace(".fits", "_convolved.fits")
                        f[1].header['XTENSION'] = 'BINTABLE'
                        # modify header parameters to new npix and pixelsize
                        f[0].header["NAXIS1"] = npix
                        f[0].header["NAXIS2"] = npix
                        f[0].header["CDELT1"] = -pixel_size / self.scale
                        f[0].header["CDELT2"] = pixel_size / self.scale
                        f[0].header["CRPIX1"] = int(f[0].header["CRPIX1"] / len(self.X) * npix)
                        f[0].header["CRPIX2"] = int(f[0].header["CRPIX2"] / len(self.X) * npix)
                        f.writeto(new_stokes_u_fits, overwrite=True)

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
                new_stokes_i_fits = self.fits_file.replace(".fits", "_convolved.fits")
                f.writeto(new_stokes_i_fits, overwrite=True, output_verify='ignore')
                new_stokes_q_fits=""
                new_stokes_u_fits=""

            #if model loaded try regridding as well
            try:
                with fits.open(self.model_file_path) as f:
                    new_image_model = interpolator(f[0].data[0, 0, :, :])(points).reshape(npix,npix)
                    f[0].data = np.zeros((f[0].data.shape[0], f[0].data.shape[1], npix, npix))
                    f[0].data[0, 0, :, :] = new_image_model
                    new_model_fits = self.model_file_path.replace(".fits", "_convolved.fits")
                    f[1].header['XTENSION'] = 'BINTABLE'
                    f[0].header["NAXIS1"] = npix
                    f[0].header["NAXIS2"] = npix
                    f[0].header["CDELT1"] = -pixel_size / self.scale
                    f[0].header["CDELT2"] = pixel_size / self.scale
                    f[0].header["CRPIX1"]=int(f[0].header["CRPIX1"]/len(self.X)*npix)
                    f[0].header["CRPIX2"]=int(f[0].header["CRPIX2"]/len(self.X)*npix)
                    f.writeto(new_model_fits, overwrite=True)
            except:
                warnings.warn("Model not regridded, probably no model loaded.",UserWarning)
                new_model_fits=""

            newImageData=ImageData(fits_file=new_stokes_i_fits,
                         uvf_file=self.uvf_file,
                         stokes_q=new_stokes_q_fits,
                         stokes_u=new_stokes_u_fits,
                         mask=new_mask,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=new_model_fits,
                         correct_rician_bias=self.correct_rician_bias,
                         difmap_path=self.difmap_path)
        else:
            #Using DIFMAP
            self.mask=new_mask
            newImageData=self.restore(-1,-1,-1,npix=npix*2,pixel_size=pixel_size,weighting=weighting,useDIFMAP=True)

        return newImageData

    def plot(self,plot_mode="stokes_i",plot_mask=False,show=True):
        #TODO Include all parameters from FitsImage here
        FitsImage(self,plot_mode="stokes_i",plot_mask=plot_mask)
        if show:
            plt.show()


    def align(self,image_data2,masked_shift=True,method="cross_correlation",auto_mask='', auto_regrid=False,useDIFMAP=True):

        if (self.Z.shape != image_data2.Z.shape) or self.degpp != image_data2.degpp:
            if auto_regrid:
                # if this is selected will automatically convolve with common beam and regrid
                print("Automatically regridding image to minimum pixelsize, smallest FOV and common beam")

                #determin common image parameters
                pixel_size=np.min([self.degpp*self.scale,image_data2.degpp*image_data2.scale])
                min_fov=np.min([self.degpp*len(self.X)*self.scale,image_data2.degpp*len(image_data2.X)*self.scale])
                npix=int(min_fov/pixel_size)

                #get common beam
                common_beam=get_common_beam([self.beam_maj,image_data2.beam_maj],
                                            [self.beam_min,image_data2.beam_min],
                                            [self.beam_pa,image_data2.beam_pa],arg="common")
                #TODO check the common beam, something fishy might be happening with the regridding

                #regrid images
                image_self = self
                image_self = image_self.regrid(npix,pixel_size,useDIFMAP=useDIFMAP)
                # convolve with common beam
                image_self = image_self.restore(common_beam[0], common_beam[1], common_beam[2], useDIFMAP=useDIFMAP)

                # same for image 2
                image_data2 = image_data2.regrid(npix, pixel_size, useDIFMAP=useDIFMAP)
                image_data2 = image_data2.restore(common_beam[0], common_beam[1], common_beam[2], useDIFMAP=useDIFMAP)


            else:
                warnings.warn("Images do not have the same npix and pixelsize, please regrid first or use auto_regrid=True.", UserWarning)
                return self

        else:
            image_self=self

        if method=="cross_correlation":
            if (np.all(image_data2.mask==False) and np.all(image_self.mask==False)) or masked_shift==False:
                shift,error,diffphase = phase_cross_correlation((image_data2.Z),(image_self.Z),upsample_factor=100)
                print('will apply shift (x,y): [{} : {}] mas'.format(-shift[1]*image_self.scale*image_self.degpp, shift[0] *image_self.scale*image_self.degpp))
                #print('register images new shift (y,x): {} px +- {}'.format(-shift, error))
            else:
                shift, _, _ = phase_cross_correlation((image_data2.Z),(image_self.Z),upsample_factor=100,reference_mask=image_data2.mask,moving_mask=image_self.mask)
                print('will apply shift (x,y): [{} : {}] mas'.format(-shift[1]*image_self.scale*image_self.degpp, shift[0]*image_self.scale*image_self.degpp))
                #print('register images new shift (y,x): {} px'.format(-shift))
        elif method=="brightest":
            #align images on brightest pixel
            #TODO
            pass
        else:
            warning.warn("Please use valid align method ('cross_correlation','brightest').",UserWarning)


        #shift shifted image
        return image_self.shift(-shift[1]*image_self.scale*image_self.degpp,shift[0]*image_self.scale*image_self.degpp,useDIFMAP=useDIFMAP)


    def restore(self,bmaj,bmin,posa,shift_x=0,shift_y=0,npix="",pixel_size="",weighting=[0,-1],useDIFMAP=True):
        """
        This allows you to restore the ImageData object with a custom beam either with DIFMAP or just the image itself
        Inputs:
            bmaj: Beam major axis (in mas)
            bmin: Beam minor axis (in mas)
            posa: Beam position angle (in deg)
        Returns:
            New ImageData object
        """

        #TODO basic sanity check if uvf file is present and if polarization is there
        if self.uvf_file=="" or useDIFMAP==False:
            #this means there is no valid .uvf file or we don't want to use DIFMAP

            print("No .uvf file attached, will do simple shift of image only")

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
                new_image_i = convolve_with_elliptical_gaussian(new_image_i, bmaj / self.scale / self.degpp/2,
                                                             bmin / self.scale / self.degpp/2, posa)
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
                    new_image_q = convolve_with_elliptical_gaussian(new_image_q,bmaj/self.scale/self.degpp/2,bmin/self.scale/self.degpp/2,posa)
                    # convert to jansky per (new) beam
                    new_image_q = Jy2JyPerBeam(new_image_q, bmaj, bmin, self.degpp * self.scale)

                input_ = np.fft.fft2(self.stokes_u)  # before it was np.fft.fftn(img)
                offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
                imgalign = np.fft.ifft2(offset_image)  # again before ifftn
                new_image_u = imgalign.real
                if not (bmaj==-1 and bmin ==-1 and posa==-1):
                    new_image_u = JyPerBeam2Jy(new_image_u, self.beam_maj, self.beam_min, self.degpp * self.scale)
                    new_image_u= convolve_with_elliptical_gaussian(new_image_u,bmaj/self.scale/self.degpp/2,bmin/self.scale/self.degpp/2,posa)
                    # convert to jansky per (new) beam
                    new_image_u = Jy2JyPerBeam(new_image_u, bmaj, bmin, self.degpp * self.scale)


            except:
                new_image_q = ""
                new_image_u = ""
                new_stokes_u_fits = ""
                new_stokes_q_fits = ""

            # if model loaded try shifting model image as well
            try:
                input_ = np.fft.fft2(
                    fits.open(self.model_file_path)[0].data[0, 0, :, :])  # before it was np.fft.fftn(img)
                offset_image = fourier_shift(input_, shift=[shift_y, shift_x])
                imgalign = np.fft.ifft2(offset_image)  # again before ifftn
                new_image_model = imgalign.real
                if not (bmaj==-1 and bmin ==-1 and posa==-1):
                    new_image_model = JyPerBeam2Jy(new_image_model, self.beam_maj, self.beam_min, self.degpp * self.scale)
                    new_image_model = convolve_with_elliptical_gaussian(new_image_model,bmaj/self.scale/self.degpp/2,bmin/self.scale/self.degpp/2,posa)
                    # convert to jansky per (new) beam
                    new_image_model = Jy2JyPerBeam(new_image_model, bmaj, bmin, self.degpp * self.scale)

                with fits.open(self.model_file_path) as f:
                    f[0].data[0, 0, :, :] = new_image_model
                    new_model_fits = self.model_file_path.replace(".fits", "_convolved.fits")
                    f[1].header['XTENSION'] = 'BINTABLE'
                    f[1].data["DELTAX"] += shift_x_deg
                    f[1].data["DELTAY"] += shift_y_deg
                    if not (bmaj == -1 and bmin == -1 and posa == -1):
                        f[0].header["BMAJ"] = bmaj / self.scale
                        f[0].header["BMIN"] = bmin / self.scale
                        f[0].header["BPA"] = posa
                    f.writeto(new_model_fits, overwrite=True)
            except:
                new_image_model = ""
                new_model_fits = ""

            #write outputs to the fitsfiles
            if self.only_stokes_i:
                # this means DIFMAP style fits image
                with fits.open(self.fits_file) as f:
                    f[0].data[0, 0, :, :] = new_image_i
                    new_stokes_i_fits = self.fits_file.replace(".fits", "_convolved.fits")
                    f[1].header['XTENSION'] = 'BINTABLE'
                    #shift model/clean components
                    f[1].data["DELTAX"] += shift_x_deg
                    f[1].data["DELTAY"] += shift_y_deg
                    if not (bmaj == -1 and bmin == -1 and posa == -1):
                        #Overwrite beam parameters in header
                        f[0].header["BMAJ"] = bmaj / self.scale
                        f[0].header["BMIN"] = bmin / self.scale
                        f[0].header["BPA"] = posa
                    f.writeto(new_stokes_i_fits, overwrite=True)

                if len(self.stokes_q) > 0:
                    with fits.open(self.stokes_q_path) as f:
                        f[0].data[0, 0, :, :] = new_image_q
                        new_stokes_q_fits = self.stokes_q_path.replace(".fits", "_convolved.fits")
                        f[1].header['XTENSION'] = 'BINTABLE'
                        # shift model/clean components
                        f[1].data["DELTAX"] += shift_x_deg
                        f[1].data["DELTAY"] += shift_y_deg
                        if not (bmaj == -1 and bmin == -1 and posa == -1):
                            # Overwrite beam parameters in header
                            f[0].header["BMAJ"] = bmaj / self.scale
                            f[0].header["BMIN"] = bmin / self.scale
                            f[0].header["BPA"] = posa
                        f.writeto(new_stokes_q_fits, overwrite=True)

                if len(self.stokes_u) > 0:
                    with fits.open(self.stokes_u_path) as f:
                        f[0].data[0, 0, :, :] = new_image_u
                        new_stokes_u_fits = self.stokes_u_path.replace(".fits", "_convolved.fits")
                        f[1].header['XTENSION'] = 'BINTABLE'
                        # shift model/clean components
                        f[1].data["DELTAX"] += shift_x_deg
                        f[1].data["DELTAY"] += shift_y_deg
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
                new_stokes_i_fits = self.fits_file.replace(".fits", "_convolved.fits")
                f.writeto(new_stokes_i_fits, overwrite=True, output_verify='ignore')

                new_stokes_q_fits=""
                new_stokes_u_fits=""

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

            if npix=="":
                npix=len(self.X)*2
            if pixel_size=="":
                pixel_size=self.degpp*self.scale

            #restore Stokes I
            new_stokes_i_fits=self.stokes_i_mod_file.replace(".mod","_convolved")

            fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa,shift_x=shift_x,shift_y=shift_y,
                    channel="i",output_dir=self.model_save_dir+"mod_files_clean",outname=new_stokes_i_fits,
                    n_pixel=npix,pixel_size=pixel_size,
                    mod_files=[self.stokes_i_mod_file],uvf_files=[self.uvf_file],weighting=weighting)

            new_stokes_i_fits+=".fits"
            #try to restore modelfit if it is ther

            try:
                new_model_fits=self.model_mod_file.replace(".mod","_convolved")

                fold_with_beam([self.fits_file], difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa, shift_x=shift_x, shift_y=shift_y,
                    channel="i", output_dir=self.model_save_dir + "mod_files_model", outname=new_model_fits,
                    n_pixel=npix, pixel_size=pixel_size,
                    mod_files=[self.model_mod_file], uvf_files=[self.uvf_file], weighting=weighting)

                new_model_fits+=".fits"
            except:
                new_model_fits=""

            #try to restore polarization as well if it is there
            try:
                new_stokes_q_fits=self.stokes_q_mod_file.replace(".mod","_convolved")
                new_stokes_u_fits=self.stokes_u_mod_file.replace(".mod","_convolved")


                fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa,shift_x=shift_x,shift_y=shift_y,
                    channel="q",output_dir=self.model_save_dir+"mod_files_q",outname=new_stokes_q_fits,
                    n_pixel=npix,pixel_size=pixel_size,
                    mod_files=[self.stokes_q_mod_file],uvf_files=[self.uvf_file],weighting=weighting)

                new_stokes_q_fits+=".fits"

                fold_with_beam([self.fits_file],difmap_path=self.difmap_path,
                    bmaj=bmaj, bmin=bmin, posa=posa, shift_x=shift_x,shift_y=shift_y,
                    channel="u",output_dir=self.model_save_dir+"mod_files_u",outname=new_stokes_u_fits,
                    n_pixel=npix,pixel_size=pixel_size,
                    mod_files=[self.stokes_u_mod_file],uvf_files=[self.uvf_file],weighting=weighting)

                new_stokes_u_fits+=".fits"

            except:
                new_stokes_q_fits=""
                new_stokes_u_fits=""

        return ImageData(fits_file=new_stokes_i_fits,
                         uvf_file=self.uvf_file,
                         stokes_q=new_stokes_q_fits,
                         stokes_u=new_stokes_u_fits,
                         mask=new_mask,
                         noise_method=self.noise_method,
                         model_save_dir=self.model_save_dir,
                         model=new_model_fits,
                         correct_rician_bias=self.correct_rician_bias,
                         difmap_path=self.difmap_path)


    def shift(self,shift_x,shift_y,npix="",pixel_size="",weighting=[0,-1],useDIFMAP=True):
        #for shifting we can just use the restore option with shift parameters, not specifying a beam
        try:
            #We can just call the restore() function without doing the restore steps
            return self.restore(-1,-1,-1,shift_x,shift_y,npix=npix,pixel_size="",weighting=weighting,useDIFMAP=useDIFMAP)
        except:
            raise Exception("No shift possible, something went wrong!")

    def masking(self, mask_type='ellipse', args=False):
        '''Mask image data object that can be used for masking the images.

        Args:
            mask_type: 'npix_x','cut_left','cut_right','radius','ellipse','flux_cut'
            args: the arguments for the mask
                'npix_x': args=[npix_x,npixy]
                'cut_left': args = cut_left
                'cut_right': args = cut_right
                'radius': args = radius
                'ellipse': args = {'e_args': [e_maj,e_min,e_pa], 'e_xoffset': xoff, 'e_yoffset': yoff}
                'flux_cut: args = flux cut

        Returns:
            masks for both images
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
            e_maj = args['e_args'][0]
            e_min = args['e_args'][1]
            e_pa = args['e_args'][2]
            e_xoffset = args['e_xoffset']
            e_yoffset = args['e_yoffset']
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
            rr, cc = ellipse(y, x, e_maj, e_min, rotation=e_pa * np.pi / 180)
            self.mask[rr, cc] = True

        if mask_type == 'flux_cut':
            flux_cut = args
            self.mask[self.Z>flux_cut*np.max(self.Z)] = True

        if mask_type == 'reset':
            self.mask=np.zeros_like(self.Z)


    def get_noise_from_shift(self,shift_factor=20):

        if self.uvf_file == "":
            warnings.warn("Shift not possible, no .uvf file attached to ImageData!", UserWarning)
            return self.noise

        size_x=len(self.X)*self.degpp*self.scale
        size_y=len(self.Y)*self.degpp*self.scale

        #shift data away to get rms
        shifted_image=self.shift(size_x*shift_factor,size_y*shift_factor)

        noise=np.std(shifted_image.Z)

        return noise







