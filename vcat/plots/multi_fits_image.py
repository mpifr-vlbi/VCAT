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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time
import sys
import pexpect
from datetime import datetime
import colormaps as cmaps
import matplotlib.ticker as ticker
from vcat.helpers import get_sigma_levs, getComponentInfo, convert_image_to_polar, wrap_evpas, closest_index, get_date, get_freq, write_mod_file
import vcat.fit_functions as ff
from vcat.kinematics import Component
from vcat.config import logger, font
from scipy.interpolate import interp1d
from vcat.plots.fits_image import FitsImage

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = font
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12


class MultiFitsImage(object):

    def __init__(self,
                 image_cube,  # ImageData object
                 mode="individual", #Choose what effect the parameters have ('individual','freq','epoch','all')
                 swap_axis=False, #If True frequency will be plotted in x-direction and time in y
                 figsize="", #define figsize
                 shared_colormap="individual", #options are 'freq', 'epoch', 'all','individual'
                 shared_colorbar=False, #if true, will plot a shared colorbar according to share_colormap setting
                 shared_sigma="max", #select which common sigma to use options: 'max','min'
                 shared_colorbar_label="", #choose custom colorbar label
                 shared_colorbar_labelsize=10, #choose labelsize of custom colorbar
                 **kwargs #additional plot params
                 ):

        super().__init__()

        self.image_cube=image_cube
        if not swap_axis:
            self.nrows, self.ncols = self.image_cube.shape
        else:
            self.ncols, self.nrows = self.image_cube.shape
        if figsize=="":
            figsize=(3*self.ncols,3*self.nrows)

        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=figsize)
        self.axes=np.atleast_2d(self.axes)

        if self.axes.shape[0]==self.ncols and self.axes.shape[1]==self.nrows:
            if not self.ncols==self.nrows and not swap_axis:
                self.axes=self.axes.T

        #read in input parameters for individual plots
        if mode=="all":
            #This means kwargs are just numbers
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                for i in range(len(self.image_cube.dates)):
                    for j in range(len(self.image_cube.freqs)):
                        kwargs[key][i, j] = value
        elif mode=="freq":
            #allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape,dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])

                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 and isinstance(value[0],(float,int)) or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j] = value
                elif len(value)==len(self.image_cube.freqs):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j]=value[j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        elif mode=="epoch":

            # allow input parameters per epoch
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 and isinstance(value[0],(float,int)) or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.image_cube.dates):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value[i]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")

        elif mode=="individual":
            # allow input parameters per frequency
            for key, value in kwargs.items():
                kwargs[key] = np.empty(image_cube.shape, dtype=object)
                kwargs[key] = np.atleast_2d(kwargs[key])
                if not isinstance(value,list) or (key in ["xlim","ylim"] and (len(value)==2 or len(value)==0)):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i, j] = value
                elif len(value) == len(self.image_cube.images) and len(value[0]) == len(self.image_cube.images[0]):
                    for i in range(len(self.image_cube.dates)):
                        for j in range(len(self.image_cube.freqs)):
                            kwargs[key][i,j] = value[i][j]
                else:
                    raise Exception(f"Please provide valid {key} parameter.")
        else:
            raise Exception("Please select valid plot mode ('individual','freq','epoch','all'")

        #check if colormap is shared between plots:
        if shared_colormap == "all":
            noises=image_cube.noises

            if shared_sigma=="max":
                index = np.unravel_index(np.argmax(noises), noises.shape)
            else:
                index = np.unravel_index(np.argmin(noises), noises.shape)

            plot=image_cube.images[index].plot(plot_mode=kwargs["plot_mode"][0,0],im_colormap=True, im_color=kwargs["im_color"][0,0],show=False)

            for i in range(len(self.image_cube.dates)):
                for j in range(len(self.image_cube.freqs)):
                    #get levs:
                    kwargs["levs"][i,j]=plot.levs
                    kwargs["levs1"][i,j]=plot.levs1
                    kwargs["levs_linpol"][i,j]=plot.levs_linpol
                    kwargs["levs1_linpol"][i,j]=plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i,j]=plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i,j]=plot.linpol_vmax
                    kwargs["fracpol_vmax"][i,j]=plot.fracpol_vmax

            col=plot.col
            plt.close(plot.fig)

        elif shared_colormap=="epoch":
            col=[]
            for i in range(len(self.image_cube.dates)):
                images=image_cube.images[i,:].flatten()
                noises = image_cube.noises[i,:].flatten()

                if shared_sigma == "max":
                    index = np.argmax(noises)
                else:
                    index = np.argmin(noises)

                plot = images[index].plot(plot_mode=kwargs["plot_mode"][i, 0], im_colormap=True, im_color=kwargs["im_color"][i,0], show=False)

                for j in range(len(self.image_cube.freqs)):
                    # get levs:
                    kwargs["levs"][i, j] = plot.levs
                    kwargs["levs1"][i, j] = plot.levs1
                    kwargs["levs_linpol"][i, j] = plot.levs_linpol
                    kwargs["levs1_linpol"][i, j] = plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i, j] = plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i, j] = plot.linpol_vmax
                    kwargs["fracpol_vmax"][i, j] = plot.fracpol_vmax

                col.append(plot.col)
                plt.close(plot.fig)


        elif shared_colormap=="freq":
            col=[]
            for j in range(len(self.image_cube.freqs)):
                images=image_cube.images[:,j].flatten()
                noises = image_cube.noises[:,j].flatten()

                if shared_sigma == "max":
                    index = np.argmax(noises)
                else:
                    index = np.argmin(noises)

                plot = images[index].plot(plot_mode=kwargs["plot_mode"][0,j], im_colormap=True, im_color=kwargs["im_color"][0,j], show=False)

                for i in range(len(self.image_cube.dates)):
                    # get levs:
                    kwargs["levs"][i, j] = plot.levs
                    kwargs["levs1"][i, j] = plot.levs1
                    kwargs["levs_linpol"][i, j] = plot.levs_linpol
                    kwargs["levs1_linpol"][i, j] = plot.levs1_linpol
                    kwargs["stokes_i_vmax"][i, j] = plot.stokes_i_vmax
                    kwargs["linpol_vmax"][i, j] = plot.linpol_vmax
                    kwargs["fracpol_vmax"][i, j] = plot.fracpol_vmax

                col.append(plot.col)
                plt.close(plot.fig)

        elif shared_colormap=="individual":
            pass
        else:
            raise Exception("Please use valid share_colormap setting ('all','epoch','freq')")


        #create FitsImage for every image
        self.plots=np.empty((self.nrows,self.ncols),dtype=object)

        for i in range(self.nrows):
            for j in range(self.ncols):

                if swap_axis:
                    image_i=j
                    image_j=i
                else:
                    image_i=i
                    image_j=j
                if self.image_cube.images[image_i,image_j]==None:
                    #turn off the plot because no data is here
                    self.axes[i,j].axis("off")
                else:
                    self.plots[i,j]=FitsImage(image_data=self.image_cube.images[image_i,image_j],
                                        stokes_i_sigma_cut=kwargs["stokes_i_sigma_cut"][image_i,image_j],
                                        plot_mode=kwargs["plot_mode"][image_i,image_j],
                                        im_colormap=kwargs["im_colormap"][image_i,image_j],
                                        contour=kwargs["contour"][image_i,image_j],
                                        contour_color=kwargs["contour_color"][image_i,image_j],
                                        contour_cmap=kwargs["contour_cmap"][image_i,image_j],
                                        contour_alpha=kwargs["contour_alpha"][image_i,image_j],
                                        contour_width=kwargs["contour_width"][image_i,image_j],
                                        im_color=kwargs["im_color"][image_i,image_j],
                                        do_colorbar=kwargs["do_colorbar"][image_i,image_j],
                                        plot_ridgeline=kwargs["plot_ridgeline"][image_i,image_j],
                                        ridgeline_color=kwargs["ridgeline_color"][image_i,image_j],
                                        plot_counter_ridgeline=kwargs["plot_counter_ridgeline"][image_i,image_j],
                                        counter_ridgeline_color=kwargs["counter_ridgeline_color"][image_i,image_j],
                                        plot_line=kwargs["plot_line"][image_i,image_j],  # Provide two points for plotting a line
                                        line_color=kwargs["line_color"][image_i,image_j],
                                        line_width=kwargs["line_width"][image_i,image_j],  # width of the line
                                        plot_beam=kwargs["plot_beam"][image_i,image_j],
                                        plot_model=kwargs["plot_model"][image_i,image_j],
                                        component_color=kwargs["component_color"][image_i,image_j],
                                        plot_comp_ids=kwargs["plot_comp_ids"][image_i,image_j],
                                        plot_clean=kwargs["plot_clean"][image_i,image_j],
                                        plot_mask=kwargs["plot_mask"][image_i,image_j],
                                        xlim=kwargs["xlim"][image_i,image_j],
                                        ylim=kwargs["ylim"][image_i,image_j],
                                        levs=kwargs["levs"][image_i,image_j],  # predefined plot levels
                                        levs1=kwargs["levs1"][image_i,image_j],  # predefined plot levels1
                                        levs_linpol=kwargs["levs_linpol"][image_i,image_j],  # predefined linpol levs
                                        levs1_linpol=kwargs["levs1_linpol"][image_i,image_j],  # predefined linepol levs1
                                        stokes_i_vmax=kwargs["stokes_i_vmax"][image_i,image_j],  # input vmax for plot
                                        fracpol_vmax=kwargs["fracpol_vmax"][image_i,image_j],  # input vmax for plot
                                        linpol_vmax=kwargs["linpol_vmax"][image_i,image_j],  # input vmax for plot
                                        plot_evpa=kwargs["plot_evpa"][image_i,image_j],
                                        evpa_width=kwargs["evpa_width"][image_i,image_j],
                                        evpa_len=kwargs["evpa_len"][image_i,image_j],
                                        fractional_evpa_distance=kwargs["fractional_evpa_distance"][image_i,image_j],
                                        lin_pol_sigma_cut=kwargs["lin_pol_sigma_cut"][image_i,image_j],
                                        evpa_distance=kwargs["evpa_distance"][image_i,image_j],
                                        rotate_evpa=kwargs["rotate_evpa"][image_i,image_j],
                                        evpa_color=kwargs["evpa_color"][image_i,image_j],
                                        title=kwargs["title"][image_i,image_j],
                                        background_color=kwargs["background_color"][image_i,image_j],
                                        fig=self.fig,
                                        ax=self.axes[i,j],
                                        font_size_axis_title=kwargs["font_size_axis_title"][image_i,image_j],
                                        font_size_axis_tick=kwargs["font_size_axis_tick"][image_i,image_j],
                                        rcparams=kwargs["rcparams"][image_i,image_j])

        #get colorbar label:
        if shared_colorbar_label == "":
            if kwargs["plot_mode"][0, 0] == "stokes_i":
                shared_colorbar_label="Flux Density [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "lin_pol":
                shared_colorbar_label="Linear Polarization [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "frac_pol":
                shared_colorbar_label="Linear Polarization Fraction"
            elif kwargs["plot_mode"][0, 0] == "spix":
                shared_colorbar_label="Spectral Index"
            elif kwargs["plot_mode"][0, 0] == "rm":
                shared_colorbar_label="Rotation Measure [rad/m^2]"
            elif kwargs["plot_mode"][0, 0] == "residual":
                shared_colorbar_label="Residual Flux Density [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "turnover":
                shared_colorbar_label="Turnover Frequency [GHz]"
            elif kwargs["plot_mode"][0, 0] == "turnover_flux":
                shared_colorbar_label = "Turnover Flux [Jy/beam]"
            elif kwargs["plot_mode"][0, 0] == "turnover_error":
                shared_colorbar_label = "Turnover Error [GHz]"
            elif kwargs["plot_mode"][0, 0] == "turnover_chisquare":
                shared_colorbar_label = r"Turnover $\chi^2$"
            else:
                shared_colorbar_label = "Flux Density [Jy/beam]"



        # check if colorbar should be plotted and is shared between plots:
        if shared_colormap == "all" and shared_colorbar:
            cbar = self.fig.colorbar(col, ax=self.axes, orientation="horizontal", fraction=0.05, pad=0.1)
            cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)
        elif shared_colormap=="epoch" and shared_colorbar:
            for i in range(len(self.image_cube.dates)):
                if swap_axis:
                    cbar = self.fig.colorbar(col[i], ax=self.axes[:, i], orientation="horizontal", fraction=0.05, pad=0.05)
                else:
                    cbar = self.fig.colorbar(col[i], ax=self.axes[i, :], orientation="vertical", fraction=0.05, pad=0.1)
                cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)
        elif shared_colormap=="freq" and shared_colorbar:
            for j in range(len(self.image_cube.freqs)):
                if swap_axis:
                    cbar = self.fig.colorbar(col[j], ax=self.axes[j, :], orientation="vertical", fraction=0.05, pad=0.05)
                else:
                    cbar = self.fig.colorbar(col[j], ax=self.axes[:, j], orientation="horizontal", fraction=0.05, pad=0.1)
                cbar.set_label(shared_colorbar_label,fontsize=shared_colorbar_labelsize)


    def export(self,name):
        if name.split(".")[-1] in ("png","jpg","jpeg","pdf","gif"):
            self.fig.savefig(name, dpi=300, bbox_inches='tight', transparent=False)
        else:
            self.fig.savefig(name+".png",dpi=300,bbox_inches="tight", transparent=False)
