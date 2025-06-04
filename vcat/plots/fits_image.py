from os import write
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.patheffects as PathEffects
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
        plot_ridgeline: Choose to plot ridgeline
        ridgeline_color: Color for ridgeline
        plot_counter_ridgeline: Choose to plot counter ridgeline
        counter_ridgeline_color= Choose color for counter ridgeline
        plot_beam: Choose whether to plot the beam or not
        beam_color: Choose beam color
        plot_gauss: Choose whether to plot modelfit components (if available in image_data)
        component_color: Choose color to plot components
        plot_clean: Choose whether to plot clean components (if available in image_data)
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
                 do_colorbar=False, #choose whether to display colorbar
                 plot_ridgeline=False, #choose whether to display the ridgeline
                 ridgeline_color="red", #choose ridgeline color
                 plot_counter_ridgeline= False,
                 counter_ridgeline_color= "red",
                 plot_line="", #Provide two points for plotting a line
                 line_color="black",
                 line_width=2, #width of the line
                 plot_beam=True, #choose whether to plot beam or not
                 beam_color="grey", #choose beam color for plot
                 plot_model=False, #choose whether to plot modelfit components
                 component_color="black", # choose component color for Gauss component
                 plot_comp_ids=False, #plot component ids
                 plot_clean=False, #choose whether to plot clean components
                 plot_mask=False, #choose whether to plot mask
                 xlim=[], #xplot limits, e.g. [5,-5]
                 ylim=[], #yplot limits
                 levs="", #predefined plot levels
                 levs1="", #predefined plot levels1
                 levs_linpol="", #predefined linpol levs
                 levs1_linpol="", #predefined linepol levs1
                 stokes_i_vmax="", #input vmax for plot
                 fracpol_vmax="", #input vmax for plot
                 linpol_vmax="", #input vmax for plot
                 plot_polar=False, #choose to plot image in polar coordinates
                 ###HERE STARTS POLARIZATION INPUT
                 plot_evpa=False, #decide whether to plot EVPA or not
                 evpa_width=1.5, #choose width of EVPA lines
                 evpa_len=-1,  # choose length of EVPA in pixels
                 evpa_border_color="", #choose color of EVPA border
                 evpa_border_width=0.5, #choose width of EVPA border
                 lin_pol_sigma_cut=3,  # choose lowest sigma contour for Lin Pol plot
                 evpa_distance=-1,  # choose distance of EVPA vectors to draw in pixels
                 fractional_evpa_distance=0.02, #if evpa_distance==-1 and evpa_len==-1, this chooses the fractional evpa distance
                 rotate_evpa=0, # rotate EVPAs by a given angle in degrees (North through East)
                 evpa_color="white", # set EVPA color for plot
                 title="", # plot title (default is date)
                 background_color="white", #background color
                 ax=None, #define custom matplotlib axes to plot on
                 fig=None, #define custom figure
                 font_size_axis_title=font_size_axis_title, #set fontsize for axis title
                 font_size_axis_tick=font_size_axis_tick, #set fontsize for axis ticks
                 rcparams={} # option to modify matplotlib look
                 ):

        super().__init__()

        mpl.rcParams.update(rcparams)
        
        #read image
        self.clean_image = image_data
        self.clean_image_file = self.clean_image.file_path
        self.model_image_file = self.clean_image.model_file_path

        #set parameters
        self.plot_mode=plot_mode
        self.name = self.clean_image.name
        self.freq = self.clean_image.freq
        X = self.clean_image.X
        Y = self.clean_image.Y
        Z = self.clean_image.Z
        self.Z=Z
        unit = self.clean_image.unit
        scale = self.clean_image.scale
        degpp = self.clean_image.degpp
        extent = self.clean_image.extent
        date=self.clean_image.date
        lin_pol=self.clean_image.lin_pol
        self.lin_pol=self.clean_image.lin_pol
        self.evpa_width=evpa_width
        # set default evpa_len if not given
        if evpa_len==-1:
            npix=len(X)
            if xlim!=[]:
                factor=(max(xlim)-min(xlim))/(max(X)-min(X))
            else:
                factor=1
            #make evpa len 2% of the image size
            self.evpa_len = factor * npix // (1/fractional_evpa_distance)
        else:
            self.evpa_len=evpa_len

        #set default evpa_distance if not provided
        if evpa_distance==-1:
            npix = len(X)
            if xlim != []:
                factor = (max(xlim) - min(xlim)) / (max(X) - min(X))
            else:
                factor = 1
            # make evpa len 2% of the image size
            self.evpa_distance = factor * npix // (1/fractional_evpa_distance)
        else:
            self.evpa_distance=evpa_distance
        self.rotate_evpa=rotate_evpa
        # Set beam parameters
        beam_maj = self.clean_image.beam_maj
        beam_min = self.clean_image.beam_min
        beam_pa = self.clean_image.beam_pa
        self.evpa_color=evpa_color
        self.evpa_border_color=evpa_border_color
        self.evpa_border_width=evpa_border_width
        self.background_color=background_color
        self.noise_method=self.clean_image.noise_method
        self.do_colorbar=do_colorbar
        self.ridgeline_color=ridgeline_color
        self.counter_ridgeline_color=counter_ridgeline_color
        self.stokes_i_vmax=stokes_i_vmax
        self.linpol_vmax=linpol_vmax
        self.fracpol_vmax=fracpol_vmax
        self.col=""
        self.lin_pol_sigma_cut=lin_pol_sigma_cut
        self.stokes_i_sigma_cut=stokes_i_sigma_cut

        #modify these parameters if polar plot is selected
        if plot_polar:
            #currently only support colormap so turn off everything else:
            plot_gauss=False
            plot_clean=False
            plot_mask=False
            #Convert Stokes I
            R, Theta, Z_polar = convert_image_to_polar(X,Y, Z)
            extent=[Theta.min(),Theta.max(),R.min(),R.max()]
            Z=Z_polar.T
            self.Z=Z

            #Convert Lin Pol
            try:
                R, Theta, lin_pol = convert_image_to_polar(X,Y, lin_pol)
                lin_pol = lin_pol.T
                self.lin_pol = lin_pol
            except:
                pass



        #plot limits
        ra_max,ra_min,dec_min,dec_max=extent

        if len(xlim) == 2:
            ra_max, ra_min = xlim
        if len(ylim) == 2:
            dec_min, dec_max = ylim

        if ax==None and fig==None:
            self.fig, self.ax = plt.subplots(1, 1)
        else:
            if fig==None:
                self.fig = plt.figure()
            else:
                self.fig = fig
            self.ax = ax


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
        if not isinstance(levs,(list,np.ndarray)) and not isinstance(levs1,(list,np.ndarray)):
            levs, levs1 = get_sigma_levs(Z,stokes_i_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_noise)

        self.levs=levs
        self.levs1=levs1
        self.levs_linpol=levs_linpol
        self.levs1_linpol=levs1_linpol

        # Image colormap
        if self.im_colormap == True and plot_mode=="stokes_i":
            self.plotColormap(Z,im_color,levs,levs1,extent,do_colorbar=self.do_colorbar)
            contour_color="white"

        if np.sum(lin_pol)!=0:
            if not isinstance(levs_linpol,(list,np.ndarray)) and not isinstance(levs1_linpol,(list,np.ndarray)):
                levs_linpol, levs1_linpol = get_sigma_levs(lin_pol, lin_pol_sigma_cut,noise_method=self.noise_method,noise=self.clean_image.difmap_pol_noise)
                self.levs_linpol = levs_linpol
                self.levs1_linpol = levs1_linpol

        if (plot_mode=="lin_pol" or plot_mode=="frac_pol") and np.sum(lin_pol)!=0:

            if plot_mode=="lin_pol":
                self.plotColormap(lin_pol,im_color,levs_linpol,self.levs1_linpol,extent,
                                  label="Linear Polarized Intensity [Jy/beam]",do_colorbar=self.do_colorbar)
            if plot_mode=="frac_pol":
                plot_lin_pol = np.array(lin_pol)
                plot_frac_pol = plot_lin_pol / np.array(self.clean_image.Z)
                plot_frac_pol = np.ma.masked_where((plot_lin_pol < self.levs1_linpol[0]) | (self.clean_image.Z<self.levs1[0]),
                                                  plot_frac_pol)

                self.plotColormap(plot_frac_pol,im_color,np.zeros(100),[0.00],extent,
                                  label="Fractional Linear Polarization",do_colorbar=self.do_colorbar)
        elif (plot_mode=="lin_pol" or plot_mode=="frac_pol"):
            logger.warning("Trying to plot polarization but no polarization loaded!")

        if plot_mode=="residual":
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.residual_map)
                Z=Z.T
            else:
                Z=self.clean_image.residual_map
            self.plotColormap(Z,im_color,levs,levs1,extent,label="Residual Flux Density [Jy/beam]", do_colorbar=self.do_colorbar)
        if plot_mode=="spix" and np.sum(self.clean_image.spix)!=0:
            self.plotColormap(self.clean_image.spix,im_color,levs,levs1,extent,label="Spectral Index", do_colorbar=self.do_colorbar)
        elif plot_mode=="spix":
            logger.warning("Trying to plot spectral index but no spectral index available!")

        if plot_mode=="rm" and np.sum(self.clean_image.rm)!=0:
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.rm)
                Z=Z.T
            else:
                Z=self.clean_image.rm

            rm=np.ma.masked_where((abs(Z) > 20000),Z)
            self.plotColormap(rm, im_color, levs, levs1, extent, label="Rotation Measure [rad/m^2]",do_colorbar=self.do_colorbar)
        elif plot_mode=="rm":
            logger.warning("Trying to plot rotation measure but no rotation measure available!")

        if plot_mode == "turnover_freq" or plot_mode=="turnover":
            if plot_polar:
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z=Z.T
            else:
                Z=self.clean_image.turnover

            to=np.ma.masked_where(Z==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Frequency [GHz]", do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_flux":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_flux)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_flux

            to=np.ma.masked_where(Z_filter==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Flux [Jy/beam]",
                              do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_error":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_error)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_error

            to=np.ma.masked_where(Z_filter==0,Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label="Turnover Error [GHz]",
                              do_colorbar=self.do_colorbar)
        if plot_mode == "turnover_chisquare":
            if plot_polar:
                _, _, Z_filter = convert_image_to_polar(X,Y, self.clean_image.turnover)
                Z_filter = Z_filter.T
                _,_,Z=convert_image_to_polar(X,Y, self.clean_image.turnover_chi_sq)
                Z=Z.T
            else:
                Z_filter=self.clean_image.turnover
                Z=self.clean_image.turnover_chi_sq

            to = np.ma.masked_where(Z_filter == 0, Z)
            self.plotColormap(to, im_color, levs, levs1, extent, label=r"Turnover $\chi^2$",
                              do_colorbar=self.do_colorbar)


        if plot_evpa and np.sum(lin_pol)!=0:
            self.plotEvpa()

        # Contour plot
        if contour == True:
            if contour_cmap=="" or contour_cmap==None:
                contour_cmap=None
            else:
                contour_color=None

            if plot_polar:
                self.ax.contour(Theta[:,0],R[0,:], Z, linewidths=contour_width, levels=levs, colors=contour_color,
                                alpha=contour_alpha,
                                cmap=contour_cmap)

            else:
                self.ax.contour(X, Y, self.Z, linewidths=contour_width, levels=levs, colors=contour_color,
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
            beam = Ellipse([ell_x, ell_y], beam_maj, beam_min,angle= -beam_pa + 90, fc=beam_color)
            self.ax.add_artist(beam)

        if title=="":
            self.ax.set_title(date + " " + "{:.0f}".format(self.freq/1e9)+" GHz", fontsize=font_size_axis_title)
        else:
            self.ax.set_title(title, fontsize=font_size_axis_title)

        #set x/y tick size
        self.ax.tick_params(axis="y",labelsize=font_size_axis_tick)
        self.ax.tick_params(axis="x",labelsize=font_size_axis_tick)

        if plot_mask:
            if plot_polar:
                _,_,mask_p=convert_image_to_polar(X,Y,self.clean_image.mask)
                mask=mask_p.T
            else:
                mask=self.clean_image.mask
            self.plotColormap(mask,"gray_r",np.zeros(100),[0.00],extent,label="Mask",do_colorbar=self.do_colorbar)


        if plot_clean:
            model_clean_df = self.clean_image.model_i

            try:
                c_x = model_clean_df["Delta_x"]
                c_y = model_clean_df["Delta_y"]
                c_flux = model_clean_df["Flux"]
            except:
                logger.warning("No clean model available!")
                c_x = []


            for j in range(len(c_x)):
                if c_flux[j] < 0.:
                    self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='red', alpha=clean_alpha,
                                 linewidth=0.2, zorder=2)
                else:
                    self.ax.plot(c_x[j] * scale, c_y[j] * scale, marker='+', color='green', alpha=clean_alpha,
                                 linewidth=0.2, zorder=2)

        # Read modelfit files in
        if plot_model:
            model_gauss_df = self.clean_image.model

            try:
                g_x = model_gauss_df["Delta_x"]
                g_y = model_gauss_df["Delta_y"]
                g_maj = model_gauss_df["Major_axis"]
                g_min = model_gauss_df["Minor_axis"]
                g_pos = model_gauss_df["PA"]
                g_flux = model_gauss_df["Flux"]
                g_date = model_gauss_df["Date"]
                g_mjd = model_gauss_df["mjd"]
                g_mjd = model_gauss_df["mjd"]
                g_year = model_gauss_df["Year"]
            except:
                logger.warning("No model available!")
                g_x = []

            for j in range(len(g_x)):
                # plot component
                for i,comp in enumerate(self.clean_image.components):
                    if comp.flux==g_flux[j]:
                        comp_id=comp.component_number

                if plot_comp_ids:
                    component_plot = self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale, id=comp_id)
                else:
                    component_plot = self.plotComponent(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], scale)


                #calculate noise at the position of the component
                try:
                    component_noise=get_noise_from_residual_map(self.clean_image.residual_map_path, g_x[j]*scale,g_y[j]*scale,np.max(X)/10,np.max(Y)/10,scale=scale)#TODO check if the /10 width works and make it changeable
                except:
                    component_noise=self.clean_image.noise

                #This is needed for the GUI
                component = Component(g_x[j], g_y[j], g_maj[j], g_min[j], g_pos[j], g_flux[j], g_date[j],
                                      g_mjd[j], g_year[j], scale=scale, freq=self.freq, noise=component_noise,
                                      beam_maj=beam_maj, beam_min=beam_min, beam_pa=beam_pa)

                self.components.append([component_plot, component])

        if plot_ridgeline:
            #plot ridgeline in image
            self.ax.plot(self.clean_image.ridgeline.X_ridg,self.clean_image.ridgeline.Y_ridg,c=self.ridgeline_color,zorder=6)

        if plot_counter_ridgeline:
            #plot counterridgeline in image
            self.ax.plot(self.clean_image.counter_ridgeline.X_ridg, self.clean_image.counter_ridgeline.Y_ridg, c=self.counter_ridgeline_color,
                         zorder=6)

        if plot_line!="":
            self.ax.plot([plot_line[0][0],plot_line[1][0]],[plot_line[0][1],plot_line[1][1]],linewidth=line_width,c=line_color,zorder=7)

        self.xmin, self.xmax = ra_min, ra_max
        self.ymin, self.ymax = dec_min, dec_max

        self.fig.subplots_adjust(left=0.13,top=0.96,right=0.93,bottom=0.2)

        # Plot look tuning
        if plot_polar:
            self.ax.set_xlim(np.min(Theta),np.max(Theta))
            self.ax.set_ylim(np.min(R),np.max(R))
            self.ax.invert_xaxis()
            self.ax.set_aspect('auto', adjustable='box', anchor='C')
            self.ax.set_xlabel("Position Angle [°]")
            self.ax.set_ylabel(f"Radius [{self.clean_image.unit}]")
        else:

            self.ax.set_aspect('equal', adjustable='box', anchor='C')
            self.ax.set_xlim(ra_min, ra_max)
            self.ax.set_ylim(dec_min, dec_max)
            self.ax.invert_xaxis()
            self.ax.set_xlabel('Relative R.A. [' + unit + ']',fontsize=font_size_axis_title)
            self.ax.set_ylabel('Relative DEC. [' + unit + ']',fontsize=font_size_axis_title)
        self.fig.tight_layout()


    def plotColormap(self,
                     Z, #2d data array to plot
                     im_color, #colormap to use
                     levs, #sigma levs output
                     levs1, #sigma levs output
                     extent, #plot lims x_min,x_max,y_min,y_max
                     label="Flux Density [Jy/beam]", #label for colorbar
                     do_colorbar=False
                     ):

        #OPTIONS for fractional polarization plot
        if label=="Fractional Linear Polarization":
            vmin=0
            if self.fracpol_vmax=="":
                vmax = np.max([0.1, np.min([0.8, np.max(Z)*.8/.7])])
                self.fracpol_vmax=vmax
            else:
                vmax=self.fracpol_vmax
            if im_color == "":
                im_color = cmaps.neon_r

            if vmax > 0.4:
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=0.4,
                                                       vmax=vmax, vmin=vmin), extent=extent)
            else:
                self.col = self.ax.imshow(Z,
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

                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax,ticks=ticks)
                    cbar.set_label(label,fontsize=self.ax.xaxis.label.get_size())
            elif vmax <=0.2:
                ticks = np.array([0.0, 0.025, 0.05, 0.75, 0.1, 0.125, 0.15, 0.175, 0.2])
                ticklabels = ["0.000", "0.025", "0.050", "0.075", "0.100", "0.125", "0.150", "0.175", "0.200"]
                final_labels=[]
                final_ticks=[]
                for tickval in ticks:
                    if vmax >= float(tickval):
                        final_ticks = np.append(final_ticks, float(tickval))
                        final_labels.append(tickval)
                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax,ticks=final_ticks)
                    cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
            else:
                if do_colorbar:
                    divider = make_axes_locatable(self.ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                    cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
            if do_colorbar:
                cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        elif label=="Linear Polarized Intensity [Jy/beam]":
            if im_color =="":
                im_color = "cubehelix_r"

            linthresh = 10.0 * self.clean_image.pol_noise
            if self.linpol_vmax=="":
                vmax = np.max([np.max(Z), linthresh])
                self.linpol_vmax=vmax
            else:
                vmax=self.linpol_vmax

            vmin = 0
            if linthresh < 0.5 * np.max([vmax, -vmin]):
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               norm=colors.SymLogNorm(linthresh=linthresh,
                                                       vmax=vmax, vmin=vmin),extent=extent)
            else:
                self.col = self.ax.imshow(Z,
                               origin='lower',
                               cmap=im_color,
                               vmax=vmax, vmin=vmin,extent=extent)

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())
        elif label=="Mask":
            if im_color=="":
                im_color="inferno"
            self.ax.imshow(Z, cmap=im_color, vmin=0, vmax=1, interpolation='none', alpha=Z.astype(float), extent=extent, origin="lower", zorder=10)

        elif label=="Residual Flux Density [Jy/beam]":
            if im_color=="":
                im_color="gray"
            self.col = self.ax.imshow(Z, cmap=im_color, extent=extent, origin="lower")

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Spectral Index":
            if im_color=="":
                im_color="hot_r"

            self.col = self.ax.imshow(Z, cmap=im_color, vmin=self.clean_image.spix_vmin, vmax=self.clean_image.spix_vmax, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Rotation Measure [rad/m^2]":
            if im_color=="":
                im_color="coolwarm"

            if self.clean_image.rm_vmin!="" and self.clean_image.rm_vmax!="":
                self.col = self.ax.imshow(Z, cmap=im_color, vmin=self.clean_image.rm_vmin, vmax=self.clean_image.rm_vmax, extent=extent, origin='lower')
            else:
                #scale up and down equally
                vmax=np.max(abs(Z))
                vmin=-vmax
                self.col = self.ax.imshow(Z, cmap=im_color, vmin=vmin, vmax=vmax, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        elif label=="Turnover Frequency [GHz]" or label=="Turnover Flux [Jy/beam]" or label=="Turnover Error [GHz]" or label=="Turnover $\Chi^2$":
            if im_color=="":
                im_color="inferno"

            self.col = self.ax.imshow(Z, cmap=im_color, extent=extent, origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

        else:
            if im_color=="":
                im_color="inferno"

            if self.stokes_i_vmax=="":
                vmax = 0.5 * np.max(Z)
                self.stokes_i_vmax=vmax
            else:
                vmax=self.stokes_i_vmax

            self.col = self.ax.imshow(Z, cmap=im_color, norm=colors.SymLogNorm(linthresh=abs(levs1[0]), linscale=0.5, vmin=levs1[0],
                                                                        vmax=vmax, base=10.), extent=extent,
                                origin='lower')

            if do_colorbar:
                divider = make_axes_locatable(self.ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(self.col, use_gridspec=True, cax=cax)
                cbar.set_label(label, fontsize=self.ax.xaxis.label.get_size())

    def plotComponent(self,x,y,maj,min,pos,scale,fillcolor="",id="",evpa=""):

        if fillcolor=="":
            # Plotting ellipses
            comp = Ellipse([x * scale, y * scale], maj * scale, min * scale,angle= -pos + 90,
                       fill=False, zorder=2, color=self.component_color, lw=0.5)
            ellipse=self.ax.add_artist(comp)
        else:
            comp = Ellipse([x * scale, y * scale], maj * scale, min * scale, angle=-pos + 90,
                           fill=True, zorder=4, facecolor=fillcolor,edgecolor=self.component_color, lw=0.5)
            ellipse = self.ax.add_artist(comp)

        #deal with point like components
        if maj==0. and min==0.:
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

        line1=self.ax.plot([maj1_x * scale, maj2_x * scale], [maj1_y * scale, maj2_y * scale], zorder=2, color=self.component_color, lw=0.5)
        line2=self.ax.plot([min1_x * scale, min2_x * scale], [min1_y * scale, min2_y * scale], zorder=2, color=self.component_color, lw=0.5)

        if id!="":
            self.ax.text(maj1_x*scale,maj1_y*scale,str(id),fontsize=10)

        if evpa!="":
            evpa1_x = x - np.sin(-np.pi / 180 * evpa) * maj * 0.75
            evpa1_y = y + np.cos(-np.pi / 180 * evpa) * maj * 0.75
            evpa2_x = x + np.sin(-np.pi / 180 * evpa) * maj * 0.75
            evpa2_y = y - np.cos(-np.pi / 180 * evpa) * maj * 0.75

            self.ax.plot([evpa1_x*scale, evpa2_x*scale],[evpa1_y*scale,evpa2_y*scale],zorder=3,color=self.component_color,lw=3)

        return [ellipse,line1,line2]

    def plotTimeline(self,min_mjd,max_mjd,current_mjd,times):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        # Leave some space at the sides (e.g., 10% of the image width)
        side_margin = 0.1 * (x_max - x_min)

        def map_mjd_to_image(mjd):
            return x_min + side_margin + (x_max - x_min - 2*side_margin) * (mjd - min_mjd) / (max_mjd - min_mjd)

        min_x = map_mjd_to_image(min_mjd)
        max_x = map_mjd_to_image(max_mjd)
        current_x = map_mjd_to_image(current_mjd)

        y_pos = y_max * 0.9  # Adjust as needed
        self.ax.plot([min_x, max_x], [y_pos, y_pos], color='black', linestyle='-', linewidth=2, label='Observation Timeline')

        # Mark the current MJD with a dot
        self.ax.scatter(current_x, y_pos, color='black', s=100, zorder=3, label=f'Current MJD: {current_mjd}')

        # Add labels for min and max MJD
        self.ax.text(min_x, y_pos*0.85, f'{min_mjd:.2f}', color='black', ha='center', fontsize=10)
        self.ax.text(max_x, y_pos*0.85, f'{max_mjd:.2f}', color='black', ha='center', fontsize=10)

        #Add increments for real observations
        for time in times:
            self.ax.vlines(x=map_mjd_to_image(time),ymin=y_pos*0.96,ymax=y_pos*1.04,color="black",linestyle="-",linewidth=2)

    def change_plot_lim(self,x_min,x_max,y_min,y_max):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def plotEvpa(self):
        evpa=self.clean_image.evpa
        evpa_len=self.evpa_len*self.clean_image.degpp*self.clean_image.scale

        stokes_i=self.Z
        # plot EVPA
        evpa = evpa + self.rotate_evpa / 180 * np.pi

        # create mask where to plot EVPA (only where stokes i and lin pol have plotted contours)
        mask = np.zeros(np.shape(stokes_i), dtype=bool)
        mask[:] = (self.lin_pol > self.levs1_linpol[0]) * (stokes_i > self.levs1[0])
        YLoc, XLoc = np.where(mask)

        y_evpa = evpa_len * np.cos(evpa[mask])
        x_evpa = evpa_len * np.sin(evpa[mask])

        SelPix = range(0, len(stokes_i), int(self.evpa_distance))

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
        evpa_lines = LineCollection(lines, colors=self.evpa_color, linewidths=self.evpa_width,zorder=5)

        if self.evpa_border_color!="":
            #evpa_border = LineCollection(lines, colors=self.evpa_border_color, linewidths=self.evpa_width+self.evpa_border_width*2,zorder=4)
            #self.ax.add_collection(evpa_border)
            evpa_lines.set_path_effects([
                PathEffects.Stroke(linewidth=self.evpa_width + self.evpa_border_width, foreground=self.evpa_border_color),  # Border
                PathEffects.Normal()  # Main line
            ])

        self.ax.add_collection(evpa_lines)


    def plotCompCollection(self,cc,freq="",epoch="",color="black",fmt="o",markersize=4,capsize=None,filter_unresolved=False,label="",
                           plot_errorbar=True):

        data=cc.get_model_profile(freq=freq,epochs=epoch,filter_unresolved=filter_unresolved)

        if plot_errorbar:
            self.ax.errorbar(data["x"],data["y"],yerr=data["y_err"],xerr=data["x_err"],fmt=fmt,markersize=markersize,
                             capsize=capsize,color=color,label=label)
        else:
            self.ax.scatter(data["x"],data["y"],marker=fmt,color=color,markersize=markersize,label=label)

    def plot_kinematic_2d_fit(self, x_min, x_max, fit_params_x, fit_params_y, color, t_mid=0, label=""):

        fit_x = np.poly1d(fit_params_x)
        fit_y = np.poly1d(fit_params_y)
        x_values = np.linspace(x_min, x_max, 1000)
        x_cor = x_values - t_mid

        self.ax.plot(fit_x(x_cor), fit_y(x_cor), color=color, label=label)

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

