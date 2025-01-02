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

#optimized draw on Agg backend
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 1000

#define some matplotlib figure parameters
mpl.rcParams['font.family'] = 'Quicksand'
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0

font_size_axis_title=13
font_size_axis_tick=12

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

    def plot_chi_square(self,uvf_files,modelfit_files,difmap_path):

        #calculate chi-squares
        chi_squares=[]
        dates=[]

        for ind,uvf in enumerate(uvf_files):
            df=getComponentInfo(modelfit_files[ind])
            freq=get_freq(modelfit_files[ind])
            write_mod_file(df,"modelfit.mod",freq,adv=True)
            chi_square=get_model_chi_square_red(uvf,"modelfit.mod",difmap_path)
            chi_squares.append(chi_square)
            dates.append(get_date(modelfit_files[ind]))
            os.system("rm -rf modelfit.mod")

        chi_squares=np.array(chi_squares)

        for ind,dat in enumerate(dates):
            #calculate decimal year
            date = datetime.strptime(dat, "%Y-%m-%d")

            # Calculate the start of the year and the start of the next year
            start_of_year = datetime(date.year, 1, 1)
            start_of_next_year = datetime(date.year + 1, 1, 1)

            # Calculate the number of days since the start of the year and total days in the year
            days_since_start_of_year = (date - start_of_year).days
            total_days_in_year = (start_of_next_year - start_of_year).days

            # Calculate the decimal year
            decimal_year = date.year + days_since_start_of_year / total_days_in_year
            dates[ind]=float(decimal_year)

        #make plot
        self.ax.plot(dates,chi_squares,color="black")
        self.ax.scatter(dates,chi_squares,color="black")

        self.ax.set_xlabel('Time [year]', fontsize=font_size_axis_title)
        self.ax.set_ylabel('Reduced Chi-Square of Modelfits', fontsize=font_size_axis_title)

        try:
            self.set_limits([np.min(dates)-0.5,np.max(dates)+0.5],
                            [np.min(chi_squares)-1,np.max(chi_squares)+1])
        except:
            pass

    def set_limits(self,x,y):
        self.ax.set_xlim(x)
        self.ax.set_ylim(y)
    def plot_linear_fit(self,x_min,x_max,slope,y0,color,label=""):
        def y(x):
            return slope*x+y0
        self.ax.plot([x_min,x_max],[y(x_min),y(x_max)],color,label=label)

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
                 im_color='', # string for matplotlib colormap
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
                 background_color="white", #background color
                 rcparams={} # option to modify matplotlib look
                 ):

        super().__init__()

        #read image
        self.clean_image = image_data
        self.clean_image_file = self.clean_image.file_path
        self.model_image_file = self.clean_image.model_file_path

        #set parameters
        self.plot_mode=plot_mode
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
        self.background_color=background_color
        self.noise_method=self.clean_image.noise_method

        #plot limits
        ra_max,ra_min,dec_min,dec_max=extent

        if len(xlim) == 2:
            ra_max, ra_min = xlim
        if len(ylim) == 2:
            dec_min, dec_max = ylim

        self.fig, self.ax = plt.subplots(1, 1)

        #set background color
        self.ax.set_facecolor(self.background_color)

        self.components=[]

        #component default color
        self.component_color = component_color

        fit_noise = True  # if True, the noise value and rms deviation will be fitted as described in the PhD-thesis of Moritz Böck (https://www.physik.uni-wuerzburg.de/fileadmin/11030400/Dissertation_Boeck.pdf); if False, the noise frome difmap will be used

        # Image colormap
        self.im_colormap = im_colormap  # if True, a image colormap will be done

        clean_alpha = 1  # float for sympol transparency

        #get sigma levs
        levs, levs1 = get_sigma_levs(Z,stokes_i_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_noise)

        # Image colormap
        if self.im_colormap == True and plot_mode=="stokes_i":
            self.plotColormap(Z,im_color,levs,levs1,extent)
            contour_color="white"


        if (plot_mode=="lin_pol" or plot_mode=="frac_pol") and np.sum(self.clean_image.lin_pol)!=0:

            levs_linpol, levs1_linpol = get_sigma_levs(self.clean_image.lin_pol, lin_pol_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_pol_noise)

            if plot_mode=="lin_pol":
                self.plotColormap(self.clean_image.lin_pol,im_color,levs_linpol,levs1_linpol,extent,
                                  label="Linear Polarized Intensity [Jy/beam]")
            if plot_mode=="frac_pol":
                plot_lin_pol = np.array(self.clean_image.lin_pol)
                plot_frac_pol = plot_lin_pol / np.array(self.clean_image.Z)
                plot_frac_pol = np.ma.masked_where((plot_lin_pol < levs1_linpol[0]) | (self.clean_image.Z<levs1[0]),
                                                  plot_frac_pol)

                self.plotColormap(plot_frac_pol,im_color,np.zeros(100),[0.00],extent,
                                  label="Fractional Linear Polarization")

        if plot_evpa and np.sum(self.clean_image.lin_pol)!=0:
            levs_linpol, levs1_linpol = get_sigma_levs(self.clean_image.lin_pol, lin_pol_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_pol_noise)
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
                    component_plot = self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale)
                    #calculate noise at the position of the component
                    try:
                        component_noise=get_noise_from_residual_map(self.clean_image.residual_map_path, g_x[j]*scale,g_y[j]*scale,np.max(X)/10,np.max(Y)/10,scale=scale)#TODO check if the /10 width works and make it changeable
                    except:
                        component_noise=self.clean_image.noise_3sigma
                    
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
                     label="Flux Density [Jy/beam]" #label for colorbar
                     ):

        #OPTIONS for fractional polarization plot
        if label=="Fractional Linear Polarization":
            vmin=0
            vmax = np.max([0.1, np.min([0.8, np.max(Z)*.8/.7])])

            if im_color == "":
                im_color = cmaps.neon_r

            if vmax > 0.4:
                col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=0.4,
                                                       vmax=vmax, vmin=vmin), extent=extent)
            else:
                col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               vmax=vmax, vmin=vmin, extent=extent)
            if vmax >= 0.4:
                ticks = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
                ticklabels = ["0.0", "", "0.1", "", "0.2", "", "0.3", "", "0.4"]
                # add appropriate ticklabels up to 0.7.
                for tickval in ["0.5", "0.6", "0.7","0.8"]:
                    if vmax >= float(tickval):
                        ticks = np.append(ticks, float(tickval))
                        ticklabels.append(tickval)
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(col, use_gridspec=True, cax=cax,ticks=ticks)
                cbar.set_label(label)
                cbar.ax.set_yticklabels(ticklabels)
            elif vmax <=0.2:
                ticks = np.array([0.0, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.175, 0.2])
                ticklabels = ["0.000", "0.025", "0.050", "0.075", "0.100", "0.125", "0.150", "0.175", "0.200"]
                final_labels=[]
                final_ticks=[]
                for tickval in ticks:
                    if vmax >= float(tickval):
                        final_ticks = np.append(final_ticks, float(tickval))
                        final_labels.append(tickval)
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(col, use_gridspec=True, cax=cax,ticks=final_ticks)
                cbar.set_label(label)
                cbar.ax.set_yticklabels(final_labels)
            else:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(col, use_gridspec=True, cax=cax)
                cbar.set_label(label)
            cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        elif label=="Linear Polarized Intensity [Jy/beam]":
            if im_color =="":
                im_color = "cubehelix_r"

            linthresh = 10.0 * levs1[0]

            vmax = np.max([np.max(Z), 10.0 * levs1[0]])
            vmin = 0
            if linthresh < 0.5 * np.max([vmax, -vmin]):
                col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=linthresh,
                                                       vmax=vmax, vmin=vmin),extent=extent)
            else:
                col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               vmax=vmax, vmin=vmin,extent=extent)
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = self.fig.colorbar(col, use_gridspec=True, cax=cax)
            cbar.set_label(label)

        else:
            if im_color=="":
                im_color="inferno"
            col = self.ax.imshow(Z, cmap=im_color, norm=colors.SymLogNorm(linthresh=abs(levs1[0]), linscale=0.5, vmin=levs1[0],
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

    def export(self,name):
        #check if name is a directory, if so create generic filename in pdf and png format
        if os.path.isdir(name):
            if name[-1]!="/":
                name+="/"
            name+=self.name+"_"+"{:.2f}".format(self.freq/1e9)+"GHz_"+self.clean_image.date+"_"+self.plot_mode
            self.fig.savefig(name+".png", dpi=300, bbox_inches='tight', transparent=False)
            self.fig.savefig(name+".pdf", dpi=300, bbox_inches='tight', transparent=False)
        else:
            if name.split(".")[-1] in ("png","jpg","jpeg","pdf","gif"):
                self.fig.savefig(name, dpi=300, bbox_inches='tight', transparent=False)
            else:
                self.fig.savefig(name+".png",dpi=300,bbox_inches="tight", transparent=False)
