import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.time import Time
import sys
from vcat import ImageData

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

        #go through image data list and extract some info
        for image in image_data_list:
            self.freqs.append(image.freq)
            self.dates.append(image.date)

        self.freqs=np.sort(np.unique(self.freqs))
        self.dates=np.sort(np.unique(self.dates))

        self.images=np.empty((len(self.dates),len(self.freqs)),dtype=object)

        for i,date in enumerate(self.dates):
            for j, freq in enumerate(self.freqs):
                for image in image_data_list:
                    if image.date==date and image.freq==freq:
                        self.images[i,j]=image

        self.shape=self.images.shape

    #print out some basic details
    def __str__(self):
        return f"ImageCube with {self.shape[1]} frequencies and {self.shape[0]} epochs."









