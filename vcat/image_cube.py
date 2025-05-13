import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.time import Time
import sys
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as opt
import numpy.ma as ma
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
from vcat.image_data import ImageData
from vcat.helpers import (get_common_beam, sort_fits_by_date_and_frequency,
                          sort_uvf_by_date_and_frequency, closest_index, func_turn,plot_pixel_fit,fit_width, coreshift_fit)
from vcat.plots.evolution_plot import EvolutionPlot
from vcat.plots.multi_fits_image import MultiFitsImage
from vcat.plots.kinematic_plot import KinematicPlot
from vcat.plots.model_image_plot import ModelImagePlot
from vcat.image_data import ImageData
from vcat.kinematics import ComponentCollection
from vcat.stacking_helpers import stack_fits, stack_pol_fits
from tqdm import tqdm
from vcat.config import uvw, plot_colors, plot_markers, plot_linestyles
from vcat.plots.jet_profile_plot import JetProfilePlot
from joblib import Parallel, delayed

#initialize logger
from vcat.config import logger

class ImageCube(object):

    """ Class to handle a multi-frequency, multi-epoch Image data set

    Attributes:
        freqs (list[float]): Frequencies of the ImageCube in Hz
        dates (list[str]): Dates included in the ImageCube
        mjds (list[float]): MJDs included in the ImageCube
        name (str): Source Name
        date_tolerance (float): Difference in days to consider the same epoch
        freq_tolerance (float): Difference in GHz to consider the same frequency
        images (list[list[ImageData]]): 2d-array of ImageData objects
        comp_collections (list[ComponentCollection]): List of Component Collections (one collection per ID)
        images_freq (list[list[float]]): 2d-array of frequencies in Hz
        images_mjd (list[list[float]]): 2d-array of MJDs
        images_majs (list[list[float]]): 2d-array of Beam Major Axis
        images_mins (list[list[float]]): 2d-array of Beam Minor Axis
        images_pas (list[list[float]]): 2d-array of Beam position angle (in degree)
        noises (list[list[float]]): 2d-array of Image noises
    """

    def __init__(self,
                 image_data_list=[], #list of ImageData objects
                 date_tolerance=1, #date tolerance to consider "simultaneous"
                 freq_tolerance=1, #frequency tolerance to consider the same,
                 new_import=True
                 ):
        """
        Initialize ImageCube class.

        Args:
            image_data_list (list[ImageData]): List of ImageData objects
            date_tolerance (float): Difference in days to consider the same epoch
            freq_tolerance (float): Difference in GHz to consider the same frequency
        """
        self.freqs=[]
        self.dates=[]
        self.mjds=[]
        self.name=""
        self.date_tolerance=date_tolerance
        self.freq_tolerance=freq_tolerance

        images=[]
        #go through image data list and extract some info
        for image in image_data_list:
            if self.name=="":
                self.name=image.name
            elif self.name != image.name and new_import:
                logger.warning(f"ImageCube setup for source {self.name} but {image.name} detected in one input file!")
            if not any(abs(num - image.freq) <= freq_tolerance*1e9 for num in self.freqs):
                self.freqs.append(image.freq)
            if not any(abs(num - image.mjd) <= date_tolerance for num in self.mjds):
                self.dates.append(image.date)
                self.mjds.append(image.mjd)
            images.append(image)

        image_data_list=images
        self.freqs=np.sort(self.freqs)
        self.dates=np.sort(self.dates)
        self.mjds=np.sort(self.mjds)

        self.images=np.empty((len(self.dates),len(self.freqs)),dtype=object)
        self.images_freq = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_mjd = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_majs = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_mins = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_pas = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.noises = np.empty((len(self.dates), len(self.freqs)), dtype=float)

        for i,mjd in enumerate(self.mjds):
            for j, freq in enumerate(self.freqs):
                for image in image_data_list:
                    if (abs(image.mjd-mjd)<=date_tolerance) and (abs(image.freq-freq)<=freq_tolerance*1e9):
                        self.images[i,j]=image
                        self.images_mjd[i,j]=image.mjd
                        self.images_freq[i,j]=image.freq
                        self.images_majs[i,j]=image.beam_maj
                        self.images_mins[i,j]=image.beam_min
                        self.images_pas[i,j]=image.beam_pa
                        self.noises[i,j]=image.noise

        self.shape=self.images.shape

        #assign component collections
        self.comp_collections=self.get_comp_collections(date_tolerance,freq_tolerance)

    #print out some basic details
    def __str__(self):
        print_freqs=[]
        for freq in self.freqs:
            print_freqs.append("{:.0f}".format(freq * 1e-9) + " GHz")
        if self.shape[1]==1 and self.shape[0]==1:
            line1 = f"ImageCube for source {self.name} with {self.shape[1]} frequency and {self.shape[0]} epoch.\n"
            line2 = f"Frequency [GHz]: " + ", ".join(print_freqs) + "\n"
            line3 = f"Epoch: " + ", ".join(self.dates)
        elif self.shape[1]==1:
            line1 = f"ImageCube for source {self.name} with {self.shape[1]} frequency and {self.shape[0]} epochs.\n"
            line2 = f"Frequency [GHz]: " + ", ".join(print_freqs) + "\n"
            line3 = f"Epochs: " + ", ".join(self.dates)
        elif self.shape[0]==1:
            line1 = f"ImageCube for source {self.name} with {self.shape[1]} frequencies and {self.shape[0]} epoch.\n"
            line2 = f"Frequencies [GHz]: " + ", ".join(print_freqs) + "\n"
            line3 = f"Epoch: " + ", ".join(self.dates)
        else:
            line1= f"ImageCube for source {self.name} with {self.shape[1]} frequencies and {self.shape[0]} epochs.\n"
            line2 = f"Frequencies [GHz]: " + ", ".join(print_freqs) + "\n"
            line3 = f"Epochs: " + ", ".join(self.dates)

        return line1+line2+line3

    def import_files(self,fits_files="", uvf_files="", stokes_q_files="", stokes_u_files="", model_fits_files="",
                     date_tolerance=1,freq_tolerance=1,**kwargs):
        """
        Function to import ImageCube directly from (.fits-)files.

        Args:
            fits_files (list[str]): List of Stokes-I or full-polarization .fits file
            uvf_files (list[str]): List of .uvf-files
            stokes_q_files (list[str]): List of Stokes Q .fits files
            stokes_u_files (list[str]): List of Stokes U .fits files
            model_fits_files (list[str]): List of modelfit .fits files (.mod files work as well, but must be sorted)
            **kwargs: Additional options as in ImageData (will be applied to all Images, e.g. 'noise_method')
        Returns:
            ImageCube object
        """
        #sort input files
        fits_files=sort_fits_by_date_and_frequency(fits_files)
        uvf_files=sort_uvf_by_date_and_frequency(uvf_files)
        stokes_q_files=sort_fits_by_date_and_frequency(stokes_q_files)
        stokes_u_files=sort_fits_by_date_and_frequency(stokes_u_files)
        try:
            model_fits_files=sort_fits_by_date_and_frequency(model_fits_files)
        except:
            logger.warning("model_fits_files need to be .fits file! Will continue assuming the .mod files are sorted by date and frequency, ascending!")

        if len(fits_files)==0 and len(model_fits_files)>0:
            fits_files=model_fits_files

        #initialize image array
        images=[]
        logger.info("Importing images:")
        for i in tqdm(range(len(fits_files)),desc="Processing"):
            fits_file = fits_files[i] if isinstance(fits_files, list) else ""
            uvf_file = uvf_files[i] if isinstance(uvf_files, list) else ""
            stokes_q_file = stokes_q_files[i] if isinstance(stokes_q_files, list) else ""
            stokes_u_file = stokes_u_files[i] if isinstance(stokes_u_files, list) else ""
            model_fits_file = model_fits_files[i] if isinstance(model_fits_files, list) else ""

            images.append(ImageData(fits_file=fits_file,uvf_file=uvf_file,stokes_q=stokes_q_file,stokes_u=stokes_u_file,model=model_fits_file,**kwargs))


        logger.info(f"Imported {len(fits_files)} images successfully.")
        #reinitialize instance
        return ImageCube(image_data_list=images,date_tolerance=date_tolerance,freq_tolerance=freq_tolerance)

    def masking(self,mode="all",mask_type="ellipse",args=False):
        """
        Function to apply mask to images included in Image Cube

        Args:
            mode (str): Choose apply mode ('all', 'freq', 'epoch'), will apply independent mask for 'all', per 'epoch' or per 'frequency'
            mask_type (str or list[str]): Mask type, if 'freq' or 'epoch' mode can be list
            args: Mask argument
        Returns:
            ImageCube with masked images
        """
        images = []

        if mode == "all":
            for image in self.images.flatten():
                image.masking(mask_type=mask_type,args=args)
                images.append(image)
        elif mode == "freq":
            for i in range(len(self.freqs)):
                # check if parameters were input per frequency or for all frequencies
                mask_type_i = mask_type[i] if isinstance(mask_type, list) else mask_type
                args_i = args[i] if isinstance(mask_type, list) else args

                image_select = self.images[:, i]
                for image in image_select:
                    image.masking(mask_type=mask_type_i, args=args_i)
                    images.append(image)
        elif mode == "epoch":
            for i in range(len(self.dates)):
                # check if parameters were input per epoch or for all epochs
                mask_type_i = mask_type[i] if isinstance(mask_type, list) else mask_type
                args_i = args[i] if isinstance(mask_type, list) else args

                image_select = self.images[i, :]
                for image in image_select:
                    image.masking(mask_type=mask_type_i,args=args_i)
                    images.append(image)
        else:
            raise Exception("Please specify valid masking mode ('all', 'epoch', 'freq')!")

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def stack(self,mode="freq", stack_linpol=False):
        """

        Args:
            mode: Select mode ("all" -> stack all images, "freq" -> stack all images from the same frequency across epochs,
            "epoch" -> stack all images from the same epoch across all frequencies)
            stack_linpol: If true, polarization will be stacked in lin_pol and EVPA instead of Q and U (not implemented yet!)
        Returns:
            new ImageCube with reduced dimension according to mode selection with stacked images
        """
        #TODO implement stack_linpol option -> how to handle new ImageData object without new fits file?
        new_fits_files=[]
        if mode=="all":
            stokes_i_fits=[]
            stokes_q_fits=[]
            stokes_u_fits=[]
            for image in self.images.flatten():
                stokes_i_fits.append(image.file_path)
                if image.stokes_q_path!="":
                    stokes_q_fits.append(image.stokes_q_path)
                if image.stokes_u_path!="":
                    stokes_u_fits.append(image.stokes_u_path)

            new_fits_i = self.images.flatten()[0].model_save_dir + "mod_files_clean/" + self.name + "_stacked.fits"

            if len(stokes_i_fits)!=len(stokes_q_fits) or len(stokes_i_fits)!=len(stokes_u_fits):
                logger.warning("Polarization data not present or invalid, will only stack Stokes I!")
                logger.info("Stacking images")
                stack_fits(fits_files=stokes_i_fits,output_file=new_fits_i)
            else:
                logger.info("Stacking images")
                stack_fits(fits_files=stokes_i_fits,stokes_q_fits=stokes_q_fits,stokes_u_fits=stokes_u_fits,
                           output_file=new_fits_i)

            new_fits_files.append(new_fits_i)

        elif mode=="freq":
            for i in range(len(self.freqs)):
                stokes_i_fits = []
                stokes_q_fits = []
                stokes_u_fits = []
                for image in self.images[:,i].flatten():
                    if image.file_path!="":
                        stokes_i_fits.append(image.file_path)
                    if image.stokes_q_path != "":
                        stokes_q_fits.append(image.stokes_q_path)
                    if image.stokes_u_path != "":
                        stokes_u_fits.append(image.stokes_u_path)

                new_fits_i = (self.images.flatten()[0].model_save_dir + "mod_files_clean/" +
                              self.name + "_" + "{:.0f}".format(self.freqs[i]*1e-9).replace(".","_") + "GHz_stacked.fits")

                if len(stokes_i_fits) != len(stokes_q_fits) or len(stokes_i_fits) != len(stokes_u_fits):
                    logger.warning("Polarization data not present or invalid, will only stack Stokes I!")
                    logger.info(f"Stacking images for {self.freqs[i]*1e-9:.1f} GHz.")
                    stack_fits(fits_files=stokes_i_fits, output_file=new_fits_i)
                else:
                    logger.info(f"Stacking images for {self.freqs[i] * 1e-9:.1f} GHz.")
                    stack_fits(fits_files=stokes_i_fits, stokes_q_fits=stokes_q_fits, stokes_u_fits=stokes_u_fits,
                               output_file=new_fits_i)

                new_fits_files.append(new_fits_i)

        elif mode=="epoch":
            for i in range(len(self.dates)):
                stokes_i_fits = []
                stokes_q_fits = []
                stokes_u_fits = []
                for image in self.images[i, :].flatten():
                    if image.file_path != "":
                        stokes_i_fits.append(image.file_path)
                    if image.stokes_q_path != "":
                        stokes_q_fits.append(image.stokes_q_path)
                    if image.stokes_u_path != "":
                        stokes_u_fits.append(image.stokes_u_path)

                new_fits_i = (self.images.flatten()[0].model_save_dir + "mod_files_clean/" +
                              self.name + "_" + self.dates[i] + "_stacked.fits")

                if len(stokes_i_fits) != len(stokes_q_fits) or len(stokes_i_fits) != len(stokes_u_fits):
                    logger.warning("Polarization data not present or invalid, will only stack Stokes I!")
                    logger.info(f"Stacking images for {self.dates[i]}.")
                    stack_fits(fits_files=stokes_i_fits, output_file=new_fits_i)
                else:
                    logger.info(f"Stacking images for {self.dates[i]}.")
                    stack_fits(fits_files=stokes_i_fits, stokes_q_fits=stokes_q_fits, stokes_u_fits=stokes_u_fits,
                               output_file=new_fits_i)

                new_fits_files.append(new_fits_i)
        else:
            raise Exception("Please specify valid stacking mode ('all', 'freq', 'epoch')")

        #create new ImageData objects from new fits files:
        images=[]
        for file in new_fits_files:
            images.append(ImageData(file,noise_method=self.images.flatten()[0].noise_method,difmap_path=self.images.flatten()[0].difmap_path))

        #return new ImageCube
        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def get_common_beam(self,mode="all",arg="common",ppe=100,tolerance=0.0001,plot_beams=False):
        """
        This function calculates the common beam from a selection of ImageData Objects within the ImageCube.

        Args:
            mode: Select mode ("all" -> one beam for all, "freq" -> gets common beam per frequency across epochs,
            "epoch" -> gets common beam per epoch across all frequencies)
            arg: Type of algorithm to use ("mean", "max", "median", "circ", "common")
            ppe: Points per Ellipse for "common" algorithm
            tolerance: Tolerance parameter for "common" algorithm
            plot_beams: Boolean to choose if a diagnostic plot of all beams and the common beam should be displayed
        Returns:
            [maj, min, pos] List with new major and minor axis and position angle
        """
        if mode=="all":
            return get_common_beam(self.images_majs.flatten(), self.images_mins.flatten(), self.images_pas.flatten(), arg=arg, ppe=ppe, tolerance=tolerance, plot_beams=plot_beams)
        elif mode=="freq":
            beams=[]
            for freq in self.freqs:
                cube=self.slice(freq_lim=[freq*1e-9-1,freq*1e-9+1])
                beam=get_common_beam(cube.images_majs.flatten(), cube.images_mins.flatten(), cube.images_pas.flatten(), arg=arg, ppe=ppe, tolerance=tolerance, plot_beams=plot_beams)
                beams.append(beam)
            return beams
        elif mode=="epoch":
            beams=[]
            for epoch in self.mjds:
                cube=self.slice(epoch_lim=[epoch-1,epoch+1])
                beam=get_common_beam(cube.images_majs.flatten(), cube.images_mins.flatten(), cube.images_pas.flatten(), arg=arg, ppe=ppe, tolerance=tolerance, plot_beams=plot_beams)
                beams.append(beam)
            return beams
        else:
            raise Exception("Please specify valid mode ('all','freq','epoch')")


    def restore(self,beam_maj=-1,beam_min=-1,beam_posa=-1,arg="common",mode="all", useDIFMAP=True,
                shift_x=0,shift_y=0,weighting=uvw,ppe=100,tolerance=0.0001,plot_beams=False):
        """
        This function allows to restore the ImageCube with a custom beam

        Args:
            beam_maj: Beam major axis to use
            beam_min: Beam minor axis to use
            beam_posa: Beam position angle to use (in degrees)
            arg: Type of algorithm to use for common beam calculation ("mean", "max", "median", "circ", "common")
            mode: Select restore mode ("all" -> applies beam to all, "freq" -> restores common beam per frequency,
            "epoch" -> restores common beam per epoch)
            useDIFMAP: Choose whether to use DIFMAP for restoring
            shift_x: Shift in mas in x-direction (list or float)
            shift_y: Shift in mas in y-direction (list or float)
            weighting: Choose weighting to use (for DIFMAP, e.g. [0,-1])
            ppe: Points per Ellipse for "common" algorithm
            tolerance: Tolerance parameter for "common" algorithm
            plot_beams: Boolean to choose if a diagnostic plot of all beams and the common beam should be displayed
        Returns:
            new ImageCube object with restored images
        """

        # get beam(s)
        if beam_maj==-1 and beam_min==-1 and beam_posa==-1:
            logger.info("Determining common beam...")
            beams=self.get_common_beam(mode=mode, arg=arg, ppe=ppe, tolerance=tolerance, plot_beams=plot_beams)
        else:
            if isinstance(beam_maj,list) and isinstance(beam_min,list) and isinstance(beam_posa,list):
                beams=[]
                for i in range(len(beam_maj)):
                    beams.append([beam_maj[i],beam_min[i],beam_posa[i]])
            else:
                beams=[beam_maj,beam_min,beam_posa]
                mode="all"

        #initialize empty array
        images = []
        logger.info("Restoring images")

        if mode=="all":
            for ind, image in enumerate(tqdm(self.images.flatten(),desc="Processing")):
                npix=len(image.X)*2
                pixel_size=image.degpp*image.scale
                new_image=image.restore(beams[0],beams[1],beams[2],shift_x=shift_x,shift_y=shift_y,npix=npix,pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP)
                images.append(new_image)
        elif mode=="freq":
            for i in range(len(self.freqs)):
                #check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x,list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y,list) else shift_y

                image_select=self.images[:,i].flatten()
                for ind2,image in enumerate(tqdm(image_select,desc="Processing")):
                    npix = len(image.X) * 2
                    pixel_size = image.degpp * image.scale
                    images.append(image.restore(beams[i][0],beams[i][1],beams[i][2],shift_x=shift_x_i,shift_y=shift_y_i,npix=npix,
                                        pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP))
        elif mode=="epoch":
            for i in tqdm(range(len(self.dates)),desc="Processing"):
                # check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x, list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y, list) else shift_y

                image_select=self.images[i,:].flatten()
                for ind2, image in enumerate(image_select):
                    npix = len(image.X) * 2
                    pixel_size = image.degpp * image.scale
                    images.append(image.restore(beams[i][0],beams[i][1],beams[i][2],shift_x=shift_x_i,shift_y=shift_y_i,npix=npix,
                                        pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP))
        else:
            raise Exception("Please specify a restore shift mode ('all', 'freq', 'epoch')")

        logger.info(f"Image modifications completed.")

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    #This function generates lightcurve-like plots to plot the evolution of flux, lin_pol etc. vs. time
    def plot_evolution(self, value="flux",freq="",show=True, savefig="",
                       colors=plot_colors, #default colors
                       markers=plot_markers, #default markers
                       linestyles=plot_linestyles,
                       evpa_pol_plot=True):

        #TODO also make ridgeline plot over several epochs possible
        if freq == "":
            freq = self.freqs
        elif isinstance(freq,(float,int)):
            freq = [freq]
        elif not isinstance(freq, list):
            raise Exception("Invalid input for 'freq'.")

        if (value=="evpa" or value=="evpa_average") and evpa_pol_plot:
            plot = EvolutionPlot(pol_plot=True)
        else:
            plot = EvolutionPlot(xlabel="MJD [days]")

        for i,f in enumerate(freq):
            values = []
            mjds = []
            ind_f=closest_index(freq,f)
            for image in self.images[:,ind_f].flatten():
                mjds.append(image.mjd)
                if value=="flux":
                    values.append(image.integrated_flux_clean)
                    ylabel="Flux Density [Jy/beam]"
                elif value=="linpol" or value=="lin_pol":
                    values.append(image.integrated_pol_flux_clean)
                    ylabel = "Linear Polarized Flux [Jy/beam]"
                elif value=="frac_pol" or value=="fracpol":
                    values.append(image.frac_pol*100)
                    ylabel = "Fractional Polarization [%]"
                elif value=="evpa" or value=="evpa_average":
                    values.append(image.evpa_average/np.pi*180)
                    ylabel = "Electric Vector Position Angle [°]"
                elif value=="noise":
                    values.append(image.noise)
                    ylabel = "Image Noise [Jy/beam]"
                elif value=="pol_noise" or value=="polnoise":
                    values.append(image.pol_noise)
                    ylabel = "Polarization Noise [Jy/beam]"
                else:
                    raise Exception("Please specify valid plot mode")


            label="{:.1f}".format(f*1e-9)+" GHz"

            if (value=="evpa" or value=="evpa_average") and evpa_pol_plot:
                plot.plotEVPAevolution(np.array(mjds),np.array(values),c=colors[i%len(colors)],marker=markers[i%len(markers)],
                                       label=label,linestyle=linestyles[i%len(linestyles)])
            else:
                plot.plotEvolution(np.array(mjds),np.array(values),c=colors[i % len(colors)],marker=markers[i % len(markers)],
                               label=label,linestyle=linestyles[i % len(linestyles)])

                plt.ylabel(ylabel)

        plt.legend()
        plt.tight_layout()

        if savefig!="":
            plt.savefig(savefig,dpi=300, bbox_inches='tight', transparent=False)

        if show:
            plt.show()

        return plot

    def plot(self, show=True, savefig="",**kwargs):
        defaults = {
            "swap_axis": False,
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
            "plot_model": False,
            "component_color": "black",
            "plot_comp_ids": False,
            "plot_clean": False,
            "plot_mask": False,
            "xlim": [],
            "ylim": [],
            "levs": "",
            "levs1": "",
            "levs_linpol": "",
            "levs1_linpol": "",
            "stokes_i_vmax": "",
            "fracpol_vmax": "",
            "linpol_vmax": "",
            "shared_colormap": "individual",  # options are 'freq', 'epoch', 'all','individual'
            "shared_colorbar": False,  # if true, will plot a shared colorbar according to share_colormap setting
            "shared_sigma": "max",  # select which common sigma to use options: 'max','min'
            "shared_colorbar_label": "",  # choose custom colorbar label
            "shared_colorbar_labelsize" : 10,  # choose labelsize of custom colorbar
            "plot_evpa": False,
            "evpa_width": 1.5,
            "evpa_len": -1,
            "lin_pol_sigma_cut": 3,
            "evpa_distance": -1,
            "fractional_evpa_distance": 0.1,
            "rotate_evpa": 0,
            "evpa_color": "white",
            "title": " ",
            "background_color": "white",
            "figsize": "",
            "font_size_axis_title": 8,
            "font_size_axis_tick": 6,
            "rcparams": {}
        }

        params = {**defaults, **kwargs}
        plot=MultiFitsImage(self,**params)

        if savefig!="":
            plot.export(savefig)
        if show:
            plt.show()

        return plot

    def regrid(self,npix="", pixel_size="",mode="all",weighting=uvw,useDIFMAP=True,mask_outside=False):
        # initialize empty array
        images = []

        logger.info("Regridding images")

        if mode=="all":
            for image in tqdm(self.images.flatten(),desc="Processing"):
                new_image=image.regrid(npix=npix,pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP,mask_outside=mask_outside)
                images.append(new_image)
        elif mode=="freq":
            for i in range(len(self.freqs)):
                # check if parameters were input per frequency or for all frequencies
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[:, i]
                for image in tqdm(image_select,desc="Processing"):
                    images.append(image.regrid(npix=npix_i, pixel_size=pixel_size_i, weighting=weighting, useDIFMAP=useDIFMAP, mask_outside=mask_outside))
        elif mode=="epoch":
            for i in tqdm(range(len(self.dates)),desc="Processing"):
                # check if parameters were input per frequency or for all frequencies
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[i, :]
                for image in image_select:
                    images.append(image.regrid(npix=npix_i, pixel_size=pixel_size_i, weighting=weighting, useDIFMAP=useDIFMAP, mask_outside=mask_outside))
        else:
            raise Exception("Please specify valid regrid mode ('all', 'epoch', 'freq')!")

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def shift(self, mode="all", shift_x=0, shift_y=0, npix="",pixel_size="",weighting=uvw,useDIFMAP=True):
        # initialize empty array
        images = []

        if mode == "all":
            for image in self.images.flatten():
                new_image = image.shift(shift_x=shift_x,shift_y=shift_y,npix=npix, pixel_size=pixel_size,
                                        weighting=weighting, useDIFMAP=useDIFMAP)
                images.append(new_image)
        elif mode == "freq":
            for i in range(len(self.freqs)):
                # check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x, list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y, list) else shift_y
                npix_i = n_pix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[:, i]
                for image in image_select:
                    images.append(
                        image.shift(shift_x=shift_x_i, shift_y=shift_y_i, npix=npix_i, pixel_size=pixel_size_i,
                                    weighting=weighting, useDIFMAP=useDIFMAP))
        elif mode == "epoch":
            for i in range(len(self.dates)):
                # check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x, list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y, list) else shift_y
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[i, :]
                for image in image_select:
                    images.append(
                        image.shift(shift_x=shift_x_i, shift_y=shift_y_i, npix=npix_i, pixel_size=pixel_size_i,
                                    weighting=weighting, useDIFMAP=useDIFMAP))
        else:
            raise Exception("Please specify valid shift mode ('all', 'epoch', 'freq')!")

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def align(self,mode="all",beam_maj=-1,beam_min=-1,beam_posa=-1,npix="",pixel_size="",
              ref_freq="",ref_epoch="",beam_arg="common",masked_shift=True,method="cross_correlation",useDIFMAP=True,ref_image="",ppe=100, tolerance=0.0001,remove_components=[]):

        # get beam(s)
        if beam_maj == -1 and beam_min == -1 and beam_posa == -1:
            beams = self.get_common_beam(mode=mode, arg=beam_arg, ppe=ppe, tolerance=tolerance, plot_beams=False)
        else:
            if isinstance(beam_maj, list) and isinstance(beam_min, list) and isinstance(beam_posa, list):
                beams = []
                for i in range(len(beam_maj)):
                    beams.append([beam_maj[i], beam_min[i], beam_posa[i]])
            else:
                beams = [beam_maj, beam_min, beam_posa]
                mode="all"

        images_new=[]
        if mode=="all":
            images=self.images.flatten()
            if ref_image=="":
                if npix=="" or pixel_size=="":
                    #find largest FOV to use for regridding
                    npixs=[]
                    pixel_sizes=[]
                    for image in images:
                        npixs=np.append(npixs,len(image.X))
                        pixel_sizes=np.append(pixel_sizes,image.degpp*image.scale)
                    fovs=npixs*pixel_sizes
                    npix=round(npixs[np.argmax(fovs)])
                    pixel_size=pixel_sizes[np.argmax(fovs)]
                else:
                    #will use custom specified npix and pixel_size
                    pass
            else:
                #use reference image
                npix=len(ref_image.X)
                pixel_size=ref_image.degpp*ref_image.scale
                beams=[ref_image.beam_maj,ref_image.beam_min,ref_image.beam_posa]

            #regrid images
            im_cube=self.regrid(npix,pixel_size,mode=mode,useDIFMAP=useDIFMAP)
            #restore images
            im_cube=im_cube.restore(beams[0],beams[1],beams[2],mode=mode,useDIFMAP=useDIFMAP)

            images=im_cube.images.flatten()

            #remove components from image before aligning
            if len(remove_components)>0:
                for j in range(len(images)):
                    images[j]=images[j].remove_component(remove_components)

            #choose reference_image (this is pretty random)
            if ref_image=="":
                ref_image=images[0]
            # align images
            for image in images:
                images_new.append(image.align(ref_image,masked_shift=masked_shift,method=method,useDIFMAP=useDIFMAP))

        elif mode=="freq":
            for i in range(len(self.freqs)):
                images=self.images[:,i].flatten()

                ref_image_i = ref_image[i] if isinstance(ref_image,list) else ref_image
                npix_i = npix[i] if isinstance(npix,list) else npix
                pixel_size_i = pixel_size[i] if isinstance(npix, list) else pixel_size

                if ref_image_i=="":
                    beam_i=beams[i]
                    if npix_i=="" or pixel_size_i=="":
                        #find largest FOV to use for regridding
                        npixs=[]
                        pixel_sizes=[]
                        for image in images:
                            npixs=np.append(npixs,len(image.X))
                            pixel_sizes=np.append(pixel_sizes,image.degpp*image.scale)
                        fovs=npixs*pixel_sizes
                        npix_i=round(npixs[np.argmax(fovs)])
                        pixel_size_i=pixel_sizes[np.argmax(fovs)]
                    else:
                        #will use custom specified npix and pixel_size
                        pass
                else:
                    #use reference image
                    npix_i=len(ref_image_i.X)
                    pixel_size_i=ref_image_i.degpp*ref_image.scale
                    beam_i=[ref_image_i.beam_maj,ref_image.beam_min,ref_image.beam_posa]

                #regrid images
                im_cube=ImageCube(images)
                im_cube=im_cube.regrid(npix_i,pixel_size_i,mode="all",useDIFMAP=useDIFMAP)
                #restore images
                im_cube=im_cube.restore(beam_i[0],beam_i[1],beam_i[2],mode="all",useDIFMAP=useDIFMAP)

                images=im_cube.images.flatten()
                #choose reference_image (this is pretty random)
                if ref_image_i=="":
                    if ref_epoch=="":
                        ref_image_i=images[0]
                    else:
                        j = closest_index(self.mjds,Time(ref_epoch).mjd)
                        ref_image_i=images[j]
                #align images
                for image in images:
                    images_new.append(image.align(ref_image_i,masked_shift=True,method=method,useDIFMAP=useDIFMAP))

        elif mode=="epoch":
            for i in range(len(self.dates)):
                images = self.images[i, :].flatten()

                ref_image_i = ref_image[i] if isinstance(ref_image, list) else ref_image
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(npix, list) else pixel_size

                if ref_image_i == "":
                    beam_i = beams[i]
                    if npix_i == "" or pixel_size_i == "":
                        # find largest FOV to use for regridding
                        npixs = []
                        pixel_sizes = []
                        for image in images:
                            npixs = np.append(npixs, len(image.X))
                            pixel_sizes = np.append(pixel_sizes, image.degpp * image.scale)
                        fovs = npixs * pixel_sizes
                        npix_i = round(npixs[np.argmax(fovs)])
                        pixel_size_i = pixel_sizes[np.argmax(fovs)]
                    else:
                        # will use custom specified npix and pixel_size
                        pass
                else:
                    # use reference image
                    npix_i = len(ref_image_i.X)
                    pixel_size_i = ref_image_i.degpp * ref_image.scale
                    beam_i = [ref_image_i.beam_maj, ref_image.beam_min, ref_image.beam_posa]

                # regrid images
                im_cube = ImageCube(images)
                im_cube = im_cube.regrid(npix_i, pixel_size_i, mode="all", useDIFMAP=useDIFMAP)
                # restore images
                im_cube = im_cube.restore(beam_i[0], beam_i[1], beam_i[2], mode="all", useDIFMAP=useDIFMAP)
                images = im_cube.images.flatten()

                # choose reference_image (this is pretty random)
                if ref_image_i == "":
                    if ref_freq == "":
                        ref_image_i = images[-1]
                    else:
                        j = closest_index(self.freqs, freq*1e9)
                        ref_image_i = images[j]
                # align images
                for image in images:
                    images_new.append(image.align(ref_image_i, masked_shift=True, method=method, useDIFMAP=useDIFMAP))
        else:
            raise Exception("Please use a valid align mode ('all', 'epoch', 'freq').")

        return ImageCube(image_data_list=images_new,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def slice(self,epoch_lim="",freq_lim=""):
        """
        This method allows you to get a slice of the given ImageCube

        Args:
            epoch_lim: [start_epoch,end_epoch] Provide start and end epoch or MJD
            freq_lim: [start_freq, end_freq] Provide start and end frequency in GHz
        Returns:
            new ImageCube with cut applied
        """

        if epoch_lim!="":
            if isinstance(epoch_lim[1], str):
                mjd_max=Time(epoch_lim[0]).mjd
            elif isinstance(epoch_lim[1], float):
                mjd_max=epoch_lim[1]
            elif isinstance(epoch_lim[1], int):
                mjd_max=epoch_lim[1]
            else:
                raise Exception("Please enter valid epoch_lim!")
            if isinstance(epoch_lim[0], str):
                mjd_min=Time(epoch_lim[0]).mjd
            elif isinstance(epoch_lim[0], float):
                mjd_min=epoch_lim[0]
            elif isinstance(epoch_lim[0], int):
                mjd_min=epoch_lim[0]
            else:
                raise Exception("Please enter valid epoch_lim!")
        else:
            mjd_min=0
            mjd_max=np.inf

        try:
            freq_min=freq_lim[0]*1e9
            freq_max=freq_lim[1]*1e9
        except:
            if freq_lim!="":
                raise Exception("Please enter valid freq_lim!")
            else:
                freq_min=0
                freq_max=np.inf

        freqs=self.images_freq.flatten()
        mjds=self.images_mjd.flatten()
        images=self.images.flatten()

        inds=np.where(np.logical_and(np.logical_and(freqs>=freq_min,freqs<=freq_max),
                      np.logical_and(mjds>=mjd_min,mjds<=mjd_max)))


        return ImageCube(image_data_list=images[inds],date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)


    def concatenate(self,ImageCube2):
        images=np.append(self.images.flatten(),ImageCube2.images.flatten())

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)


    def removeFreq(self, freq="",window=1.):
        """
        This method allows you to remove a particular frequency.

        Args:
            freq: List of frequencies to remove
            window: Window in GHz to consider around center freq
        Returns:
            new ImageCube
        """
        window = float(window)
        cubes=[]
        if freq!="":
            if isinstance(freq, float) or isinstance(freq, int):
                freq=[freq]
            if isinstance(freq,list):
                freq=np.sort(freq)
                for ind,frequency in enumerate(freq):
                    if ind==0:
                        cube=self.slice(freq_lim=[0,frequency-window])
                    else:
                        cube=self.slice(freq_lim=[freq[ind-1]+window,frequency-window])
                    cubes.append(cube)

                cubes.append(self.slice(freq_lim=[freq[-1] + window, np.inf]))

            start_cube=cubes[0]
            for i in range(1,len(cubes)):
                start_cube=start_cube.concatenate(cubes[i])

        return start_cube


    def removeEpoch(self, epoch="",window=1.):
        """
        This method allows you to remove a particular epoch.

        Args:
            epoch: List of epochs to remove
            window: Days to consider around the epoch
        Returns:
            new ImageCube
        """
        window=float(window)

        cubes = []
        if epoch != "":
            if isinstance(epoch, float) or isinstance(epoch, int):
                epoch = [epoch]
            if isinstance(epoch, str):
                epoch = [epoch]
            if isinstance(epoch, list):
                epoch = np.sort(epoch)
                for ind, ep in enumerate(epoch):
                    if isinstance(ep, str):
                        ep=Time(ep).mjd
                        epoch[ind]=ep
                    if ind == 0:
                        cube = self.slice(epoch_lim=[0, ep - window])
                    else:
                        cube = self.slice(epoch_lim=[epoch[ind - 1] + window, ep - window])
                    cubes.append(cube)

                cubes.append(self.slice(epoch_lim=[float(epoch[-1]) + window, np.inf]))

            start_cube = cubes[0]
            for i in range(1, len(cubes)):
                start_cube = start_cube.concatenate(cubes[i])

        return start_cube

    def get_spectral_index_map(self,freq1,freq2,ref_image="",epoch="",spix_vmin=-3,spix_vmax=5,sigma_lim=3,plot=False):
        #TODO implement fitting spix across more than two frequencies

        #TODO basic check if images are aligned and same pixels if not, align automatically

        if isinstance(epoch, list):
            epochs=epoch
        elif epoch=="":
            epochs=self.dates
        else:
            epochs=[epoch]

        spec_ind_maps=[]
        for epoch in epochs:
            i=closest_index(self.mjds,Time(epoch).mjd)
            images=self.images[i,:].flatten()

            #find images to use
            image1=images[closest_index(self.freqs,freq1*1e9)]
            image2=images[closest_index(self.freqs,freq2*1e9)]

            #filter according to sigma cut
            spix1=image1.Z*(image1.Z>image1.noise*sigma_lim)*(image2.Z>image2.noise*sigma_lim)
            spix2=image2.Z*(image2.Z>image2.noise*sigma_lim)*(image1.Z>image1.noise*sigma_lim)

            spix1[spix1==0] = image1.noise*sigma_lim
            spix2[spix2==0] = image2.noise*sigma_lim

            a = np.log10(spix2/spix1)/np.log10(freq2/freq1)

            logger.info('Spectral index max(alpha)={} - min(alpha)={}\nCutoff {}<alpha<{}'.format(ma.amax(a),ma.amin(a),spix_vmin,spix_vmax))

            a[a<spix_vmin]=spix_vmin
            a[a>spix_vmax]=spix_vmax
            a[spix2==image2.noise*sigma_lim] = spix_vmin

            # TODO maybe it makes sense to introduce a new SpixData Class here? The current solution is a bit hacky, but it works
            if ref_image=="":
                ref_image=image2
            image_copy=ref_image.copy()
            image_copy.spix=a
            image_copy.is_spix=True
            image_copy.spix_vmin=spix_vmin
            image_copy.spix_vmax=spix_vmax
            if plot:
                image_copy.plot(plot_mode="spix",im_colormap=True,do_colorbar=True)

            spec_ind_maps.append(image_copy)

        return ImageCube(image_data_list=spec_ind_maps,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def get_images(self,freq="",epoch=""):
        if isinstance(epoch,str) and epoch!="":
            mjd=Time(epoch).mjd
        elif isinstance(epoch,float) or isinstance(epoch,int):
            mjd=epoch

        if epoch=="" and freq=="":
            return self.images
        elif epoch=="":
            freq_ind = closest_index(self.freqs, freq * 1e9)
            return self.images[:,freq_ind]
        elif freq=="":
            time_ind = closest_index(self.mjds,mjd)
            return self.images[time_ind,:]
        else:
            time_ind=closest_index(self.mjds,mjd)
            freq_ind=closest_index(self.freqs,freq*1e9)
            return self.images[time_ind,freq_ind]

    def get_rm_map(self,freq1,freq2,epoch="",sigma_lim=3,rm_vmin="",rm_vmax="",sigma_lim_pol=5,plot=False):
        #TODO get RM map across more than 2 frequencies by fitting

        # TODO basic check if images are aligned and same pixels if not, align automatically

        if isinstance(epoch, list):
            epochs=epoch
        elif epoch=="":
            epochs=self.dates
        else:
            epochs=[epoch]

        rm_maps=[]
        for epoch in epochs:
            i=closest_index(self.mjds,Time(epoch).mjd)
            images=self.images[i,:].flatten()

            #find images to use
            image1=images[closest_index(self.freqs,freq1*1e9)]
            image2=images[closest_index(self.freqs,freq2*1e9)]

            # filter according to sigma cut
            evpa1 = (image1.evpa * (image1.Z > image1.noise * sigma_lim) * (image1.lin_pol > image1.pol_noise * sigma_lim_pol)
                     *(image2.Z > image2.noise * sigma_lim) * (image2.lin_pol > image2.pol_noise * sigma_lim_pol))
            evpa2 = (image2.evpa * (image2.Z > image2.noise * sigma_lim) * (image2.lin_pol > image2.pol_noise * sigma_lim_pol)
                     * (image1.Z > image1.noise * sigma_lim) * (image1.lin_pol > image1.pol_noise * sigma_lim_pol))

            evpa1[evpa1 == 0] = 0
            evpa2[evpa2 == 0] = 1000 #for masked areas will create incredibly high RM that will be filtered later


            #calculate wavelengths
            lam1=c.si.value/image1.freq
            lam2=c.si.value/image2.freq

            # calculate rotation measure
            rm=(evpa2-evpa1)/(lam2**2-lam1**2)


            #calculate intrinsic EVPA
            evpa0=(evpa1*lam2**2-evpa2*lam1**2)/(lam2**2-lam1**2)

            # TODO maybe it makes sense to introduce a new RMData Class here? The current solution is a bit hacky, but it works
            image_copy = image2.copy()
            image_copy.rm = rm #write rotation measure to image
            image_copy.evpa = evpa0 #write intrinsic evpa to evpa
            image_copy.is_rm = True
            image_copy.rm_vmin=rm_vmin
            image_copy.rm_vmax=rm_vmax
            if plot:
                image_copy.plot(plot_mode="rm",im_colormap=True,do_colorbar=True)

            rm_maps.append(image_copy)

        return ImageCube(image_data_list=rm_maps,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def get_turnover_map(self,epoch="",ref_image="",sigma_lim=10,max_feval=1000000,alphat=2.5,specific_pixel=(-1,-1),limit_freq=True):
        #Largely imported from Luca Ricci's Turnover frequency code
        #TODO basic error handling to check if the files are aligned and regridded and restored.
        func_turn_fixed= partial(func_turn, alphat=alphat)


        if isinstance(epoch, list):
            epochs=epoch
        elif epoch=="":
            epochs=self.dates
        else:
            epochs=[epoch]

        frequencies=np.array(self.freqs)*1e-9 #Frequencies in GHz

        final_images=[]

        for epoch in epochs:
            i = closest_index(self.mjds, Time(epoch).mjd)
            images = self.images[i, :].flatten()

            #initialize result arrays
            turnover = np.zeros_like(images[0].Z)
            turnover_flux = np.zeros_like(images[0].Z)
            chi_square = np.zeros_like(images[0].Z)
            error_map = np.zeros_like(images[0].Z)

            lowest_freq = frequencies[0]
            highest_freq = frequencies[-1]

            for i in range(len(images[0].Z)):
                for j in range(len(images[0].Z[0])):
                    brightness = []
                    err_brightness = []

                    for image in images:
                        if image.Z[i,j] > image.noise * sigma_lim:
                            brightness.append(image.Z[i,j])
                            err_brightness.append(image.Z[i,j]*image.error)

                    if len(brightness) == len(images):
                        try:
                            popt, pcov = curve_fit(func_turn_fixed, frequencies, brightness, sigma=err_brightness,
                                                   maxfev=max_feval)
                            perr = np.sqrt(np.diag(pcov))
                            x_vals = np.linspace(lowest_freq,highest_freq,1000)
                            y_vals = func_turn_fixed(x_vals, *popt)
                            peak_idx = np.argmax(y_vals)
                            turnover_freq = x_vals[peak_idx]
                            peak_brightness = y_vals[peak_idx]

                            if (lowest_freq +1 <= turnover_freq <= highest_freq -1) or not limit_freq:
                                turnover[i,j] = turnover_freq
                                turnover_flux[i,j] = peak_brightness

                                # Calculate error on turnover frequency
                                popt_plus = popt + perr  # Parameters with added errors
                                popt_minus = popt - perr  # Parameters with subtracted errors

                                # Perturbed turnover frequencies
                                y_vals_plus = func_turn_fixed(x_vals, *popt_plus)
                                y_vals_minus = func_turn_fixed(x_vals, *popt_minus)
                                turnover_freq_plus = x_vals[np.argmax(y_vals_plus)]
                                turnover_freq_minus = x_vals[np.argmax(y_vals_minus)]

                                # Error as average absolute difference
                                error_map[i, j] = 0.5 * (abs(turnover_freq_plus - turnover_freq) + abs(
                                    turnover_freq_minus - turnover_freq))
                            else:
                                turnover[i,j] = 0
                            chi_square[i,j] = np.sum(((np.array(brightness) - func_turn_fixed(np.array(frequencies), *popt)) / np.array(err_brightness))**2)
                            # Plot specific pixel

                            if (i, j) == specific_pixel:
                                fitted_func = func_turn_fixed(np.array(frequencies), *popt)
                                plot_pixel_fit(frequencies, brightness, err_brightness, fitted_func, specific_pixel,
                                               popt, turnover_freq)

                        except:
                            continue

            # TODO maybe it makes sense to introduce a new TurnoverData Class here? The current solution is a bit hacky, but it works
            if ref_image=="":
                image_copy = images[-1].copy()
            else:
                image_copy=ref_image
            image_copy.is_turnover = True
            image_copy.turnover = turnover
            image_copy.turnover_flux = turnover_flux
            image_copy.turnover_error = error_map
            image_copy.turnover_chi_sq = chi_square

            final_images.append(image_copy)

        return ImageCube(image_data_list=final_images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def rotate(self,angle,mode="all",useDIFMAP=True):
        images = []

        if mode == "all":
            for image in self.images.flatten():
                new_image = image.rotate(angle,useDIFMAP=useDIFMAP)
                images.append(new_image)
        elif mode == "freq":
            for i in range(len(self.freqs)):
                # check if parameters were input per frequency or for all frequencies
                angle_i = angle[i] if isinstance(angle, list) else angle
                image_select = self.images[:, i]
                for image in image_select:
                    new_image=image.rotate(angle_i,useDIFMAP=useDIFMAP)
                    images.append(new_image)
        elif mode == "epoch":
            for i in range(len(self.dates)):
                # check if parameters were input per frequency or for all frequencies
                angle_i = angle[i] if isinstance(angle, list) else angle

                image_select = self.images[i, :]
                for image in image_select:
                    new_image = image.rotate(angle_i, useDIFMAP=useDIFMAP)
                    images.append(new_image)
        else:
            raise Exception("Please specify valid rotate mode ('all', 'epoch', 'freq')!")

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def center(self,mode="stokes_i",useDIFMAP=True):
        images=[]

        for image in self.images.flatten():
            images.append(image.center(mode=mode,useDIFMAP=useDIFMAP))

        return ImageCube(image_data_list=images,date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance,
                         new_import=False)

    def get_ridgeline(self,mode="all",**kwargs):
        """
        Function to call get_ridgeline for all images in the ImageCube to fit a ridgeline for all images.

        Args:
            mode (str): Choose apply mode ('all', 'freq', 'epoch', 'individual'), will apply independent mask for 'all', per 'epoch' or per 'frequency'
            **kwargs: Options  for get_ridgeline() on ImageCube, need to passed as individual values (mode='all') or lists (mode='frequency' or 'epoch) or 2d-arrays (mode='individual')
        """
        kwargs=self.format_kwargs(kwargs,mode)

        for i in range(len(self.images)):
            for j in range(len(self.images[0])):
                single_kwargs={key: value[i][j] for key, value in kwargs.items()}
                self.images[i][j].get_ridgeline(**single_kwargs)

    def get_core_comp_collection(self):
        for cc in self.comp_collections:
            if cc.components[0].is_core:
                return cc

        raise Exception(f"No component collection with id {comp_id} found.")

    def get_comp_collection(self,comp_id):
        for cc in self.comp_collections:
            if cc.ids[0,0]==comp_id:
                return cc

        raise Exception(f"No component collection with id {comp_id} found.")

    def get_comp_collections(self,date_tolerance=1,freq_tolerance=1):
        #find available component ids
        comp_ids=[]
        for image in self.images.flatten():
            try:
                for comp in image.components:
                    comp_ids.append(comp.component_number)
            except:
                pass

        comp_ids=np.unique(comp_ids)

        #create a ComponentCollection for every component ID
        component_collections=[]
        for id in comp_ids:
            comps=[]
            for image in self.images.flatten():
                for comp in image.components:
                    if comp.component_number==id and comp.component_number!=-1:
                        comps.append(comp)

            if id!=-1:
                component_collections.append(ComponentCollection(components=comps,name="Component "+str(id),date_tolerance=date_tolerance,freq_tolerance=freq_tolerance))

        return component_collections

    def import_component_association(self,file):

        df=pd.read_csv(file)

        for i in range(len(self.images)):
            for j in range(len(self.images[0])):
                image = self.images[i, j]
                for k, comp in enumerate(image.components):
                    x = comp.x
                    y = comp.y
                    flux = comp.flux
                    mjd = comp.mjd
                    freq = comp.freq

                    # Find the closest component in the dataframe
                    df['distance'] = np.sqrt(
                        (df['x'] - x) ** 2 +
                        (df['y'] - y) ** 2 +
                        (df['freq'] - freq) **2 +
                        (df['mjd'] - mjd) ** 2
                    )
                    closest_row = df.loc[df['distance'].idxmin()]

                    # Assign new component number and is_core with type casting
                    self.images[i, j].components[k].is_core = bool(closest_row["is_core"])
                    self.images[i, j].components[k].component_number = int(closest_row["component_number"])

        self.update_comp_collections()

    def update_comp_collections(self):
        self.comp_collections=self.get_comp_collections()

    def fit_comp_spectrum(self,id,epoch="",fluxerr=False,fit_free_ssa=False,plot=False):

        if epoch=="":
            epochs=Time(self.dates).decimalyear
        elif isinstance(epoch,str):
            epochs=Time(np.array(epoch)).decimalyear
        elif not isinstance(epoch, list):
            raise Exception("Invalid input for 'epoch'.")


        cc=self.get_comp_collection(id)
        fit=cc.fit_comp_spectrum(epochs=epochs,fluxerr=fluxerr,fit_free_ssa=fit_free_ssa)

        for i in range(len(epochs)):
            if plot:
                plot=KinematicPlot()
                plot.plot_spectrum(cc, "black", epochs=epochs[i])
                plot.plot_spectral_fit(fit[i])
                plot.set_scale("log", "log")
                plt.show()

        return fit

    def fit_coreshift(self,ids,epoch="",k_r="",plot=False,combine_epoch=True,combine_comp=True):

        if epoch=="":
            epochs=Time(self.dates).decimalyear
        elif isinstance(epoch,str):
            epochs=Time(np.array(epoch)).decimalyear
        elif not isinstance(epoch, list):
            raise Exception("Invalid input for 'epoch'.")

        if isinstance(ids, int):
            ids=[ids]
        elif not isinstance(ids, list):
            raise Exception("Please provide valid id (int or list[int])")

        fits=[]

        for i in ids:
            cc=self.get_comp_collection(i)
            fit=cc.get_coreshift(epochs=epochs,k_r=k_r)
            fits.append(fit)

        freq_to_fit = []
        coreshift_to_fit = []
        coreshift_err_to_fit = []
        for j in range(len(epochs)):
            for i in range(len(ids)):
                freq_to_fit=np.concatenate([fits[i][j]["freqs"],freq_to_fit])
                coreshift_to_fit=np.concatenate([fits[i][j]["coreshifts"],coreshift_to_fit])
                coreshift_err_to_fit=np.concatenate([fits[i][j]["coreshift_err"],coreshift_err_to_fit])
                ref_freq=fits[i][j]["ref_freq"]

                if not combine_comp and not combine_epoch:
                    #do the fit
                    fit=coreshift_fit(freq_to_fit,coreshift_to_fit,coreshift_err_to_fit,ref_freq,k_r=k_r,print=True)

                    if plot:
                        plot=KinematicPlot()
                        plot.plot_coreshift_fit(fit)
                        plt.show()

                    freq_to_fit = []
                    coreshift_to_fit = []
                    coreshift_err_to_fit = []

            if not combine_epoch and combine_comp:
                # do the fit
                fit = coreshift_fit(freq_to_fit, coreshift_to_fit, coreshift_err_to_fit,ref_freq,k_r=k_r,print=True)
                if plot:
                    plot = KinematicPlot()
                    plot.plot_coreshift_fit(fit)
                    plt.show()

                freq_to_fit = []
                coreshift_to_fit = []
                coreshift_err_to_fit = []

        if combine_epoch and combine_comp:
            # do the fit
            fit = coreshift_fit(freq_to_fit, coreshift_to_fit, coreshift_err_to_fit,ref_freq,k_r=k_r,print=True)

            if plot:
                plot = KinematicPlot()
                plot.plot_coreshift_fit(fit)
                plt.show()

        return fit

    def get_ridgeline_profile(self,value="width",counter_ridgeline=False,freq="",epoch=""):
        """
        This function returns the ridgeline profiles combined over several epochs and frequencies

        Args:
            value (str): Value to extract ('flux', 'width', etc.)
            counter_ridgeline (bool): Choose whether to extract info for ridgeline (default, False) or counter_ridgeline (True)
            freq: Option to filter frequencies
            epoch: Option to filter epochs

        Returns:
            dist, value, value_err
        """

        #get filtered images
        images=self.get_images(freq=freq,epoch=epoch)

        #initialize arrays
        ridgelines=[]
        dists=[]
        values=[]
        value_errs=[]

        #retrieve ridgelines
        for image in images.flatten():
            if counter_ridgeline==False:
                ridgelines.append(image.ridgeline)
            else:
                ridgelines.append(image.counter_ridgeline)

        for ridgeline in ridgelines:
            #we will use the first ridgeline as reference (can be any ridgeline)
            ref_ridgeline=ridgelines[0]
            #calculate distance between ridgeline start to reference ridgeline
            delta_x=ridgeline.X_ridg[0] - ref_ridgeline.X_ridg[0]
            delta_y=ridgeline.Y_ridg[0] - ref_ridgeline.Y_ridg[0]
            delta=np.sqrt(delta_x**2+delta_y**2)

            #check if we need to subtract or add
            ridg_dist=np.array(ridgeline.dist)
            if delta_x*(ridgeline.X_ridg[-1]-ridgeline.X_ridg[0])+delta_y*(ridgeline.Y_ridg[-1]-ridgeline.Y_ridg[0])<0:
                #delta and jet direction anti-parallel
                ridg_dist-=delta
            else:
                #delta and jet direction parallel
                ridg_dist+=delta

            dists=np.concatenate((dists,ridg_dist))

            #extract values
            if value=="width":
                values=np.concatenate((values,ridgeline.width))
                value_errs=np.concatenate((value_errs,ridgeline.width_err))
            elif value=="open_angle":
                values=np.concatenate((values,ridgeline.open_angle))
                value_errs=np.concatenate((value_errs,ridgeline.open_angle_err))
            elif value=="intensity" or value=="flux":
                values=np.concatenate((values,ridgeline.intensity))
                value_errs=np.concatenate((value_errs,ridgeline.intensity_err))
            else:
                raise Exception(f"Invalid value '{value}' for 'value' parameter (allowed: 'width', 'open_angle', 'intensity')")


        #now we re-reference everything so that the ridgeline distance starts at 0
        if len(dists)>0:
            dists-=np.min(dists)

        return dists, values, value_errs

    def get_model_profile(self,value="maj",id="",freq="",epoch="",show=False,core_position=False,plot=False,filter_unresolved=False):
        """
        Get information from the model components vs. distance from

        Args:
            value (str): Choose which parameter to retrieve ("flux","maj", "tb")
            id (list[int]): Choose which components to use (default: all)
            freq: Frequencies to use
            epoch: Epochs to use
            show (bool): Choose to display a plot
            core_position: Provide reference core position (will be used to calculate distance for every component)
            plot (bool): Choose whether to generate plot
            filter_unresolved (bool): Choose whether to filter out unresolved components

        Returns:
            distance, values, value_err
        """

        if id=="":
            #do it for all components
            ccs=self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs=[]
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        #extract data
        values = []
        dists = []
        value_errs = []

        #get reference core position
        if not core_position:
            try:
                core=self.images[0,0].get_core_component()
                core_position=[core.x*core.scale,core.y*core.scale]
            except:
                core_position=[0,0]

        for cc in ccs:
            info=cc.get_model_profile(freq=freq,epochs=epoch,core_position=core_position, filter_unresolved=filter_unresolved)
            try:
                values+=info[value]
                if value=="maj" or value=="flux":
                    value_errs+=info[value+"_err"]
            except:
                raise Exception("Invalid 'value' parameter (use 'maj', 'flux' or 'tb').")
            dists=np.concatenate((dists,info["dist"]))

        if plot:
            plt.errorbar(dists, values,yerr=value_errs,fmt=".")
            plt.xlabel("Distance from Core [mas]")
            if value == "maj":
                plt.ylabel("Component Size [mas]")
            elif value == "flux":
                plt.ylabel("Flux Density [Jy]")
            else:
                plt.ylabel("Brightness Temperature [K]")
        if show:
            plt.show()

        return dists, values, value_errs

    def get_average_component(self,id="",freq="",epoch="",weighted=True):

        if id=="":
            #do it for all components
            ccs=self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs=[]
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        average_comps=[]
        for cc in ccs:
            average_comps.append(cc.get_average_component(freq=freq,epochs=epoch,weighted=weighted))

        return average_comps

    def fit_collimation_profile(self,freq="",epoch="",id="",method="model",jet="Jet",fit_type='brokenPowerlaw',x0_bpl=[0.3,0,1,2],x0_pl=[0.1,1],
                                plot_data=True,plot_fit=True,plot="",show=False,filter_unresolved=False,label="",color=plot_colors[0],marker=plot_markers[0],core_position=[0,0]):
        """
        Function to fit a collimation profile to the jet/counterjet

        Args:
            method (str): Method to use for collimation profile ('model' to use model components, 'ridgeline' to use ridgeline fit)
            jet (str): Choose whether to do Jet ('Jet'), Counterjet ('Cjet') or both ('Twin')
            fit_type (str): Choose fit_type to use ('brokenPowerlaw' or 'Powerlaw')
            x0_bpl (list[float]): Start values for broken power law fit
            x0_pl (list[float]): Start values for powerlaw fit
            plot_data (bool): Choose whether to plot the fitted data
            plot_fit (bool): Choose whether to plot the fit
            plot (JetProfilePlot): Pass JetProfilePlot to add plots, default will create a new one
            show (bool): Choose whether to show the plot
            filter_unresolved (bool): Choose whether to filter out unresolved components
            label (str): Label for the fitted data/fit
            color (str): Plot color
            marker (str): Plot marker
            core_position (list[float]): Core position in image coordinates (mas) for distance calculation
        Returns:
            plot (JetProfilePlot)

        """

        fit_fail_jet=False
        fit_fail_counterjet=False

        if method=="model":
            #jet info
            dists, widths, width_errs = self.get_model_profile("maj",id=id,freq=freq,epoch=epoch,core_position=core_position, filter_unresolved=filter_unresolved)

            #TODO get counter jet info
            cdists = []
            cwidths = []
            cwidth_errs = []

        elif method=="ridgeline":

            #jet info
            dists, widths, width_errs = self.get_ridgeline_profile(value="width",counter_ridgeline=False,freq=freq,epoch=epoch)

            #counterjet info
            cdists, cwidths, cwidth_errs = self.get_ridgeline_profile(value="width",counter_ridgeline=True,freq=freq,epoch=epoch)

        else:
            raise Exception("Please specify valid 'method' for fit_collimation_profile ('model', 'ridgeline').")

        if jet=="Jet" or jet=="Twin":
            try:
                beta, sd_beta, chi2, out = fit_width(dists, widths, width_err=width_errs, dist_err=False,
                                                     fit_type=fit_type,x0_bpl=x0_bpl,x0_pl=x0_pl)
            except:
                logger.warning("Collimation fit did not work for jet!")
                fit_fail_jet=True

        if jet=="CJet" or jet=="Twin":
            try:
                cbeta, csd_beta, cchi2, cout = fit_width(cdists, cwidths, width_err=cwidth_errs, dist_err=False,
                                                     fit_type=fit_type,x0_bpl=x0_bpl,x0_pl=x0_pl)
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
                    plot.plot_fit(x, fit_type, beta, sd_beta, chi2, "Jet", color, label=label)
            if jet=="CJet" or jet=="Twin":
                if not fit_fail_counterjet:
                    plot.plot_fit(x, fit_type, cbeta, csd_beta, cchi2, "CJet", color, label=label)

        if show:
            plot.plot_legend()
            plt.show()

        return plot

    def plot_component_evolution(self,value="flux",id="",freq="",show=True,colors=plot_colors,markers=plot_markers,
                                 evpa_pol_plot=True,plot_errors=True):
        if freq=="":
            freq=self.freqs
        elif not isinstance(freq, list):
            raise Exception("Invalid input for 'freq'.")

        if id=="":
            #do it for all components
            ccs=self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs=[]
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        for fr in freq:
            # One plot per frequency with all components
            if (value=="evpa" or value=="EVPA") and evpa_pol_plot:
                plot=KinematicPlot(pol_plot=True)
            else:
                plot = KinematicPlot()
            years = []
            for ind, cc in enumerate(ccs):
                color_ind = ind % len(colors)
                color = colors[color_ind]
                marker_ind = ind % len(markers)
                marker = markers[marker_ind]

                if value=="flux":
                    plot.plot_fluxs(cc, color=color,marker=marker,plot_errors=plot_errors)
                elif value=="tb":
                    plot.plot_tbs(cc, color=color,marker=marker)
                elif value=="dist":
                    plot.plot_kinematics(cc, color=color,marker=marker,plot_errors=plot_errors)
                elif value=="pos" or value=="PA":
                    plot.plot_pas(cc, color=color,marker=marker)
                elif value=="lin_pol" or value=="linpol":
                    plot.plot_linpol(cc, color=color,marker=marker)
                elif value=="evpa" or value=="EVPA":
                    plot.plot_evpa(cc, color=color,marker=marker)
                    years=np.concatenate((years,cc.year.flatten()))
                elif value=="maj":
                    plot.plot_maj(cc, color=color,marker=marker,plot_errors=plot_errors)
                elif value=="min":
                    plot.plot_min(cc, color=color,marker=marker,plot_errors=plot_errors)
                elif value=="theta":
                    plot.plot_theta(cc, color=color,marker=marker,plot_errors=plot_errors)
                elif value=="fracpol" or value=="frac_pol":
                    plot.plot_fracpol(cc, color=color,marker=marker)

            #set plot lims for polar plot according to lowest and highest year
            if (value=="evpa" or value=="EVPA") and evpa_pol_plot:
                years_range = max(years) - min(years)
                plot.ax.set_rmin(min(years) - 0.05 * years_range)
                plot.ax.set_rmax(max(years) + 0.05 * years_range)


            plt.legend()
            if show:
                plt.show()

    def plot_components(self,id="",freq="",epoch="",show=False,xlim=[10,-10],ylim=[-10,10],colors="",fmts=[""],markersize=4,labels=[""],
                        filter_unresolved=False,capsize=None,plot_errorbar=True):
        if id=="":
            #do it for all components
            ccs=self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs=[]
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        if colors=="":
            colors=plot_colors

        plot=ModelImagePlot(xlim=xlim,ylim=ylim)

        for i,cc in enumerate(ccs):
            color=colors[i % len(colors)]
            fmt=fmts[i % len(fmts)]
            label=labels[i%len(labels)]
            if label=="":
                label=cc.name
            plot.plotCompCollection(cc,freq=freq,epoch=epoch,color=color,fmt=fmt,markersize=markersize,capsize=capsize,
                                    filter_unresolved=filter_unresolved,label=label,plot_errorbar=plot_errorbar)

        if show:
            plot.show()

    def plot_ridgelines(self,show=False,xlim=[10,-10],ylim=[-10,10],colormap="viridis",vmin="",vmax="",linewidths=[2],labels=[""]):

        plot = ModelImagePlot(xlim=xlim, ylim=ylim)

        if vmin=="":
            vmin=np.min(self.mjds)
        if vmax=="":
            vmax=np.max(self.mjds)

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(colormap)

        for ind,image in enumerate(self.images.flatten()):
            ridgeline=image.ridgeline
            linewidth=linewidths[ind%len(linewidths)]
            label=labels[ind%len(linewidths)]
            if label=="":
                label=image.date
            plot.plotRidgeline(ridgeline,color=cmap(norm(image.mjd)),label=label,linewidth=linewidth)

        if show:
            plot.show()

    def get_speed(self,id="",freq="",order=1,show_plot=False, colors=plot_colors):

        if freq=="":
            freq=self.freqs
        elif not isinstance(freq, list):
            raise Exception("Invalid input for 'freq'.")

        if id=="":
            #do it for all components
            ccs=self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs=[]
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        fits=[]
        for fr in freq:
            #One plot per frequency with all components
            plot = KinematicPlot()
            for ind,cc in enumerate(ccs):

                fit=cc.get_speed(freqs=fr,order=order)

                for f in fit:
                    tmin=np.min(cc.year.flatten())
                    tmax=np.max(cc.year.flatten())


                    ind = ind % len(colors)
                    color=colors[ind]

                    plot.plot_kinematics(cc,color=color)
                    plot.plot_kinematic_fit(tmin-0.1*(tmax-tmin),tmax+0.1*(tmax-tmin),
                                         f["linear_fit"],color=color,label=cc.name,t_mid=f["t_mid"])

                    fits.append(f)

            if show_plot:
                plt.legend()
                plt.show()

        return fits

    def get_speed2d(self,id="",order=1,freq="",show_plot=False,plot_trajectory=False,colors=plot_colors):

        if freq == "":
            freq = self.freqs
        elif not isinstance(epoch, list):
            raise Exception("Invalid input for 'freq'.")

        if id == "":
            # do it for all components
            ccs = self.get_comp_collections(date_tolerance=self.date_tolerance, freq_tolerance=self.freq_tolerance)
        elif isinstance(id, list):
            ccs = []
            for i in id:
                ccs.append(self.get_comp_collection(i))
        else:
            raise Exception("Invalid input for 'id'.")

        fits = []
        for fr in freq:
            # One plot per frequency with all components
            if plot_trajectory:
                plot = ModelImagePlot()
            else:
                plot = KinematicPlot()
            for ind, cc in enumerate(ccs):

                fit_x,fit_y=cc.get_speed2d(freqs=fr,order=order)

                tmin = np.min(cc.year.flatten())
                tmax = np.max(cc.year.flatten())

                ind = ind % len(colors)
                color = colors[ind]

                if plot_trajectory:
                    plot.plot_kinematic_2d_fit(tmin-0.1*(tmax-tmin),tmax+0.1*(tmax-tmin),
                                             fit_x[0]["linear_fit"],fit_y[0]["linear_fit"],
                                               color=color,label=cc.name,t_mid=fit_x[0]["t_mid"])
                else:
                    plot.plot_kinematics(cc,color=color)
                    plot.plot_kinematic_2d_fit(tmin-0.1*(tmax-tmin),tmax+0.1*(tmax-tmin),
                                             fit_x[0]["linear_fit"],fit_y[0]["linear_fit"],
                                               color=color,label=cc.name,t_mid=fit_x[0]["t_mid"])

                fits.append([fit_x[0],fit_y[0]])
            if show_plot:
                plt.legend()
                plt.show()

        return fits

    def movie(self,plot_mode="stokes_i",freq="",noise="max",n_frames=500,interval=50,
              start_mjd="",end_mjd="",dpi=300,fps=20,save="",plot_components=False,fill_components=False,
              plot_timeline=True, component_cmap="hot_r",title="",**kwargs):
        """
        Function to create movies from image cube

        Args:
            plot_mode (str): Choose plot mode ('stokes_i','lin_pol','frac_pol')
            freq (float or list[float]): Choose frequencies in GHz to create movie
            noise (str): Choose which common noise level to use ('min' or 'max')
            n_frames (int): Number of frames
            interval (float): Interval in milliseconds between frames
            start_mjd (float): Start MJD of the movie
            end_mjd (float): End MJD of the movie
            dpi (int): Choose resolution in dpi
            save (str): Choose name of the movie file
            plot_components (bool): Choose whether to animate modelfit components (correct assignment should be done before!)
            fill_components (bool): If true, will fill components with a colormap based on their flux density
            plot_timeline (bool): Choose whether to plot a timeline
            component_cmap (str): Matplotlib colormap for 'fill_components' option.
            title (str): Choose Plot title (default: MJD)
            **kwargs: Plot options known from plot() function
        """

        #TODO sanity check if all images have same dimensions, otherwise it will crash
        if freq=="":
            freq=[f*1e-9 for f in self.freqs]
        elif isinstance(freq, (float,int)):
            freq=[freq]
        elif isinstance(freq,list):
            pass
        else:
            raise Exception("Please enter valid 'freq' value.")

        if save=="":
            save=[]
            for f in freq:
                save.append(f"movie_{f:.0f}GHz.mp4")
        elif isinstance(save, str):
            save_ar=[]
            if len(freq)>1:
                for f in freq:
                    save_ar.append(save+f"_{f:.0f}GHz.mp4")
                save=save_ar
            else:
                save=[save]
        elif not isinstance(save, list):
            raise Exception("Please enter valid 'save' value.")


        for index,f in enumerate(freq):
            # create figure environment to plot the data on.
            fig, ax = plt.subplots()

            ind=closest_index(self.freqs,f*1e9)
            image_datas=self.images[:,ind].flatten()

            images=[]
            lin_pols=[]
            evpas=[]
            times=[]
            #Generate interpolator function
            for image in image_datas:
                images.append(image.Z)
                lin_pols.append(image.lin_pol)
                evpas.append(image.evpa)
                times.append(image.mjd)

            grid=(times,np.arange(len(images[0])),np.arange(len(images[0][0])))

            #Stokes I
            interp_i = RegularGridInterpolator(grid, images, method='linear', bounds_error=False,
                                                  fill_value=None)
            #Lin Pol
            interp_linpol = RegularGridInterpolator(grid, lin_pols, method='linear', bounds_error=False,
                                               fill_value=None)

            #check for EVPA rotations >90 and wrap them (otherwise the EVPAs might be spinning around like crazy)
            for i, evpa in enumerate(evpas):
                if i>0:
                    for k in range(len(evpa)):
                        for j in range(len(evpa[0])):
                            if evpa[k][j]-evpas[i-1][k][j]>np.pi/2:
                                for l in range(i,len(evpas)):
                                    evpas[l][k][j]-=np.pi
                            if evpa[k][j]-evpas[i-1][k][j]<-np.pi/2:
                                for l in range(i,len(evpas)):
                                    evpas[l][k][j]+=np.pi

            #EVPA
            interp_evpa = RegularGridInterpolator(grid, evpas, method='linear', bounds_error=False,
                                               fill_value=None)

            if noise=="max":
                im_ind=np.argmax(self.noises[:,ind].flatten())
            if noise=="min":
                im_ind=np.argmin(self.noises[:,ind].flatten())

            ref_image=self.images[:,ind].flatten()[im_ind]

            #get levs
            plot=ref_image.plot(plot_mode=plot_mode,show=False,**kwargs)
            plt.close()
            levs_linpol = plot.levs_linpol
            levs1_linpol = plot.levs1_linpol
            levs = plot.levs
            levs1 = plot.levs1
            linpol_vmax = plot.linpol_vmax
            fracpol_vmax = plot.fracpol_vmax

            if start_mjd=="":
                start_mjd=np.min(self.images_mjd[:,ind].flatten())
            if end_mjd=="":
                end_mjd=np.max(self.images_mjd[:,ind].flatten())

            mjd_frames=np.linspace(start_mjd,end_mjd,n_frames)
            logger.info("Creating movie")
            progress_bar=tqdm(total=n_frames,desc="Processing")

            def update(frame):
                progress_bar.update(1)
                ax.cla()
                #modify ref_image to interpolated values
                current_mjd=mjd_frames[frame]
                X,Y=np.meshgrid(np.arange(len(images[0])),np.arange(len(images[0][0])),indexing="ij")
                query_points=np.array([np.full_like(X,current_mjd,dtype=float),X,Y]).T.reshape(-1,3)
                ref_image.Z=interp_i(query_points).reshape(len(images[0]), len(images[0][0])).T
                ref_image.stokes_i = ref_image.Z
                ref_image.lin_pol = interp_linpol(query_points).reshape(len(images[0]), len(images[0][0])).T
                ref_image.evpa = interp_evpa(query_points).reshape(len(images[0]), len(images[0][0])).T

                #plot the ref_image
                plot=ref_image.plot(plot_mode=plot_mode,fig=fig, ax=ax, show=False, title=f"MJD: {current_mjd:.0f}",
                               levs=levs,levs1=levs1,levs_linpol=levs_linpol,levs1_linpol=levs1_linpol,
                                linpol_vmax=linpol_vmax, fracpol_vmax=fracpol_vmax,**kwargs)

                #plot_components if necessary:
                if plot_components:
                    for cc in self.get_comp_collections(date_tolerance=self.date_tolerance,freq_tolerance=self.freq_tolerance):
                        #interpolate component
                        comp_interpolated=cc.interpolate(mjd=current_mjd,freq=f)
                        #try plotting it (comp_interpolated could be None if mjd is out of range)
                        try:
                            #check if we want to colorcode the component flux
                            if fill_components:
                                colormap=cm.get_cmap(component_cmap)
                                flux_color=colormap(colors.Normalize(vmin=np.min(cc.fluxs),vmax=np.max(cc.fluxs))(comp_interpolated.flux))
                            else:
                                flux_color=""
                            #plot the interpolated component
                            plot.plotComponent(comp_interpolated.x,comp_interpolated.y,comp_interpolated.maj,comp_interpolated.min,
                                               comp_interpolated.pos,comp_interpolated.scale,fillcolor=flux_color)
                        except:
                            pass

                #plot timeline
                if plot_timeline:
                    plot.plotTimeline(start_mjd,end_mjd,current_mjd,times)

            #create animation
            ani = animation.FuncAnimation(fig, update, frames=n_frames,interval=interval, blit=False)


            ani.save(save[index],writer="ffmpeg",dpi=dpi,fps=round(1/interval*1000))
            logger.info(f"Movie for {f:.0f}GHz exported as '{save[index]}'")

    def format_kwargs(self,kwargs,mode):

        # read in input parameters for individual plots
        if mode == "all":
            # This means kwargs are just numbers
            for key, value in kwargs.items():
                kwargs[key] = np.empty(self.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                for i in range(len(self.dates)):
                    for j in range(len(self.freqs)):
                        kwargs[key][i, j] = value
        elif mode == "freq":
            # allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(self.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                if not isinstance(value, list):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.freqs):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value[j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        elif mode == "epoch":

            # allow input parameters per epoch
            for key, value in kwargs.items():
                kwargs[key] = np.empty(self.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value, list):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.dates):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value[i]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")

        elif mode == "individual":
            # allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(self.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value, list):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.images) and len(value[0]) == len(self.images[0]):
                    for i in range(len(self.dates)):
                        for j in range(len(self.freqs)):
                            kwargs[key][i, j] = value[i][j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        else:
            raise Exception("Please select valid mode ('individual','freq','epoch','all'")

        return kwargs
