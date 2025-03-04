import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.time import Time
import sys
from vcat.image_data import ImageData
from vcat.helpers import get_common_beam
from vcat.graph_generator import MultiFitsImage
from vcat.image_data import ImageData
import matplotlib.pyplot as plt
from vcat.stacking_helpers import stack_fits, stack_pol_fits
import warnings

class ImageCube(object):

    """ Class to handle a multi-frequency, multi-epoch Image data set
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
                 image_data_list=[], #list of ImageData objects
                 date_tolerance=1, #date tolerance to consider "simultaneous" #TODO
                 ):
        self.freqs=[]
        self.dates=[]
        self.mjds=[]
        self.name=""

        images=[]
        #go through image data list and extract some info
        for image in image_data_list:
            skip=False
            if self.name=="":
                self.name=image.name
            elif self.name != image.name:
                warnings.warn(f"ImageCube setup for source {self.name} but {image.name} detected in one input file, will skip it.",UserWarning)
                skip=True
            if not skip:
                self.freqs.append(image.freq)
                self.dates.append(image.date)
                self.mjds.append(image.mjd)
                images.append(image)

        image_data_list=images
        self.freqs=np.sort(np.unique(self.freqs))
        self.dates=np.sort(np.unique(self.dates))
        self.mjds=np.sort(np.unique(self.mjds))

        self.images=np.empty((len(self.dates),len(self.freqs)),dtype=object)
        self.images_freq = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_mjd = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_majs = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_mins = np.empty((len(self.dates), len(self.freqs)), dtype=float)
        self.images_pas = np.empty((len(self.dates), len(self.freqs)), dtype=float)

        for i,date in enumerate(self.dates):
            for j, freq in enumerate(self.freqs):
                for image in image_data_list:
                    if image.date==date and image.freq==freq:
                        self.images[i,j]=image
                        self.images_mjd[i,j]=image.mjd
                        self.images_freq[i,j]=image.freq
                        self.images_majs[i,j]=image.beam_maj
                        self.images_mins[i,j]=image.beam_min
                        self.images_pas[i,j]=image.beam_pa

        self.shape=self.images.shape

    #print out some basic details
    def __str__(self):
        return f"ImageCube with {self.shape[1]} frequencies and {self.shape[0]} epochs."

    def import_files(self,fits_files="", uvf_files="", stokes_q_files="", stokes_u_files="", model_fits_files=""):
        #TODO create ImageData object with files
        pass

    def mask(self):
        #TODO Not sure if we really need this, should probably be done on every image individually....
        pass

    def stack(self,mode="freq", stack_linpol=False):
        """
        Args:
            mode: Select mode ("all" -> stack all images, "freq" -> stack all images from the same frequency across epochs,
            "epoch" -> stack all images from the same epoch across all frequencies)
            stack_linpol: If true, polarization will be stacked in lin_pol and EVPA instead of Q and U
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
                warnings.warn("Polarization data not present or invalid, will only stack Stokes I!",UserWarning)
                stack_fits(fits_files=stokes_i_fits,output_file=new_fits_i)
            else:
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
                              self.name + "_" + "{:.0f}".format(self.freqs[i]).replace(".","_") + "GHz_stacked.fits")

                if len(stokes_i_fits) != len(stokes_q_fits) or len(stokes_i_fits) != len(stokes_u_fits):
                    warnings.warn("Polarization data not present or invalid, will only stack Stokes I!", UserWarning)
                    stack_fits(fits_files=stokes_i_fits, output_file=new_fits_i)
                else:
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
                    warnings.warn("Polarization data not present or invalid, will only stack Stokes I!", UserWarning)
                    stack_fits(fits_files=stokes_i_fits, output_file=new_fits_i)
                else:
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
        return ImageCube(image_data_list=images)

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
                shift_x=0,shift_y=0,npix="",pixel_size="",weighting=[0,-1],ppe=100,tolerance=0.0001,plot_beams=False):
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
            npix: Number of pixels in one image direction (list or float)
            pixel_size: pixel size in mas (list or float)
            weighting: Choose weighting to use (for DIFMAP, e.g. [0,-1])
            ppe: Points per Ellipse for "common" algorithm
            tolerance: Tolerance parameter for "common" algorithm
            plot_beams: Boolean to choose if a diagnostic plot of all beams and the common beam should be displayed
        Returns:
            new ImageCube object with restored images
        """

        # get beam(s)
        if beam_maj==-1 and beam_min==-1 and beam_posa==-1:
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

        if mode=="all":
            for image in self.images.flatten():
                new_image=image.restore(beams[0],beams[1],beams[2],shift_x=shift_x,shift_y=shift_y,npix=npix,
                                        pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP)
                images.append(new_image)
        elif mode=="freq":
            for i in range(len(self.freqs)):
                #check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x,list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y,list) else shift_y
                npix_i = n_pix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select=self.images[:,i]
                for image in image_select:
                    images.append(image.restore(beams[i][0],beams[i][1],beams[i][2],shift_x=shift_x_i,shift_y=shift_y_i,npix=npix_i,
                                        pixel_size=pixel_size_i,weighting=weighting,useDIFMAP=useDIFMAP))
        elif mode=="epoch":
            for i in range(len(self.dates)):
                # check if parameters were input per frequency or for all frequencies
                shift_x_i = shift_x[i] if isinstance(shift_x, list) else shift_x
                shift_y_i = shift_y[i] if isinstance(shift_y, list) else shift_y
                npix_i = n_pix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select=self.images[i,:]
                for image in image_select:
                    images.append(image.restore(beams[i][0],beams[i][1],beams[i][2],shift_x=shift_x_i,shift_y=shift_y_i,npix=npix_i,
                                        pixel_size=pixel_size_i,weighting=weighting,useDIFMAP=useDIFMAP))
        else:
            raise Exception("Please specify a restore shift mode ('all', 'freq', 'epoch')")

        return ImageCube(image_data_list=images)

    def plot(self, show=True, **kwargs):
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
            "plot_beam": True,
            "overplot_gauss": False,
            "component_color": "black",
            "overplot_clean": False,
            "plot_mask": False,
            "xlim": [],
            "ylim": [],
            "plot_evpa": False,
            "evpa_width": 2,
            "evpa_len": 8,
            "lin_pol_sigma_cut": 3,
            "evpa_distance": 10,
            "rotate_evpa": 0,
            "evpa_color": "white",
            "titles": [],
            "background_color": "white",
            "figsize": "",
            "font_size_axis_title": 8,
            "font_size_axis_tick": 6,
            "rcparams": {}
        }

        params = {**defaults, **kwargs}
        MultiFitsImage(self,**params)

        if show:
            plt.show()

    def regrid(self,npix="", pixel_size="",mode="all",weighting=[0,-1],useDIFMAP=True,mask_outside=False):
        # initialize empty array
        images = []

        if mode=="all":
            for image in self.images.flatten():
                new_image=image.regrid(npix=npix,pixel_size=pixel_size,weighting=weighting,useDIFMAP=useDIFMAP,mask_outside=mask_outside)
                images.append(new_image)
        elif mode=="freq":
            for i in range(len(self.freqs)):
                # check if parameters were input per frequency or for all frequencies
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[:, i]
                for image in image_select:
                    images.append(image.regrid(npix=npix_i, pixel_size=pixel_size_i, weighting=weighting, useDIFMAP=useDIFMAP, mask_outside=mask_outside))
        elif mode=="epoch":
            for i in range(len(self.dates)):
                # check if parameters were input per frequency or for all frequencies
                npix_i = npix[i] if isinstance(npix, list) else npix
                pixel_size_i = pixel_size[i] if isinstance(pixel_size, list) else pixel_size

                image_select = self.images[i, :]
                for image in image_select:
                    images.append(image.regrid(npix=npix_i, pixel_size=pixel_size_i, weighting=weighting, useDIFMAP=useDIFMAP, mask_outside=mask_outside))
        else:
            raise Exception("Please specify valid regrid mode ('all', 'epoch', 'freq')!")

        return ImageCube(image_data_list=images)

    def shift(self, mode="all", shift_x=0, shift_y=0, npix="",pixel_size="",weighting=[0,-1],useDIFMAP=True):
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

        return ImageCube(image_data_list=images)

    def align(self,mode="epoch",beam_maj=-1,beam_min=-1,beam_posa=-1,npix="",pixel_size="",beam_arg="common",method="cross_correlation",useDIFMAP=True,ref_image="",ppe=100, tolerance=0.0001):

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
            #choose reference_image (this is pretty random)
            if ref_image=="":
                ref_image=images[0]
            # align images
            for image in images:
                images_new.append(image.align(ref_image,masked_shift=True,method=method,useDIFMAP=useDIFMAP))

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
                im_cube=im_cube.regrid(npix_i,pixel_size_i,mode=mode,useDIFMAP=useDIFMAP)
                #restore images
                im_cube=im_cube.restore(beam_i[0],beam_i[1],beam_i[2],mode=mode,useDIFMAP=useDIFMAP)

                images=im_cube.images.flatten()
                #choose reference_image (this is pretty random)
                if ref_image_i=="":
                    ref_image_i=images[0]
                # align images
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
                im_cube = im_cube.regrid(npix_i, pixel_size_i, mode=mode, useDIFMAP=useDIFMAP)
                # restore images
                im_cube = im_cube.restore(beam_i[0], beam_i[1], beam_i[2], mode=mode, useDIFMAP=useDIFMAP)

                images = im_cube.images.flatten()
                # choose reference_image (this is pretty random)
                if ref_image_i == "":
                    ref_image_i = images[-1] #use highest frequency by default
                # align images
                for image in images:
                    images_new.append(image.align(ref_image_i, masked_shift=True, method=method, useDIFMAP=useDIFMAP))
        else:
            raise Exception("Please use a valid align mode ('all', 'epoch', 'freq').")

        return ImageCube(image_data_list=images_new)

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


        return ImageCube(image_data_list=images[inds])


    def concatenate(self,ImageCube2):
        images=np.append(self.images.flatten(),ImageCube2.images.flatten())

        return ImageCube(image_data_list=images)


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

    def get_spectral_index_map(self):
        #TODO get spectral index map
        pass

    def get_rm_map(self):
        #TODO get RM map
        pass