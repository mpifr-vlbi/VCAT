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

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = font
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12


class ModelImagePlot(object):
    def __init__(self,xlim=[10,-10],ylim=[-10,10],xlabel="Relative R.A. [mas]",ylabel="Relative Dec. [mas]",font_size_axis_title=10):

        super().__init__()
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_xlabel(xlabel, fontsize=font_size_axis_title)
        self.ax.set_ylabel(ylabel, fontsize=font_size_axis_title)

    def plotCompCollection(self,cc,freq="",epoch="",color="black",fmt="o",markersize=4,capsize=None,filter_unresolved=False,label="",
                           plot_errorbar=True):

        data=cc.get_model_profile(freq=freq,epochs=epoch,filter_unresolved=filter_unresolved)

        if plot_errorbar:
            self.ax.errorbar(data["x"],data["y"],yerr=data["y_err"],xerr=data["x_err"],fmt=fmt,markersize=markersize,
                             capsize=capsize,color=color,label=label)
        else:
            self.ax.scatter(data["x"],data["y"],marker=fmt,color=color,markersize=markersize,label=label)

    def plotRidgeline(self,ridgeline,color="black",label="",linewidth=2):

        self.ax.plot(ridgeline.X_ridg,ridgeline.Y_ridg,color=color,lw=linewidth,label=label)

    def show(self):
        plt.legend()
        plt.tight_layout()
        plt.show()
