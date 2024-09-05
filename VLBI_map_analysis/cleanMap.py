#!/usr/bin/env python
# -*- coding: utf-8 -*-
#VCAT/modules/cleanMap.py

"""Provides the cleanMap class to plot VLBI clean map and models.

This module allows the user to handle multiple clean map fits files and to
create plots of the clean map, including color and/or contours.

Examples:
    >>> from VCAT.VLBI_map_analysis.modules.cleanMap import *
    >>> cmap = CleanMap('example_data/NGC1052_U_map.fits')
    >>> cmap.plotMap(plot_mod='example_data/NGC1052_U_model.fits')
    (<Figure size 1920x1362 with 1 Axes>, <AxesSubplot: xlabel='RA [mas]', ylabel='DEC [mas]'>)

"""

import numpy as np
import VCAT.VLBI_map_analysis.modules.fit_functions as ff
from VCAT.VLBI_map_analysis.modules.plot_functions import *
from VCAT.VLBI_map_analysis.modules.jet_calculus import *

plt.style.use('pubstyle')
plt.ioff()
class CleanMap:
    '''Organize and plot VLBI clean maps.

    Here, a clean map is loaded, either directly the img from the fits or using
    the Pyhton module ehtim, which has to be instlaled separately. Then a map
    can be plotted in color scales and/or as contour. Optinal a model components
    fits file can be loaded and the model components be plotted on top of the map.

    Attributes:
        map: The input fits file for the clean map.
        ccomp: Loading clean components, too.
        load_eht: Loas clean maps using ehtim library.
        map: The clean map fits file.
        head: The header of the fits file.
        comp: The clean component list.
        cmap: The image loaded from the fits file or with ehtim.
        cmaph: The clean component header.
        modFile: The model components fits file.
        modh: The header of the model components fits.

    Examples:
        >>> cmap = CleanMap('example_data/NGC1052_U_map.fits')
        >>> cmap.plotMap(plot_mod='example_data/NGC1052_U_model.fits')
        (<Figure size 1920x1362 with 1 Axes>, <AxesSubplot: xlabel='RA [mas]', ylabel='DEC [mas]'>)
    '''
    def __init__(self,mapFile, ccomp = False, load_eht = False):
        """Initializes the instance for the clean map.

        Args:
            map: The input fits file for the clean map.
            ccomp: Loading clean components, too.
            load_eht: Loas clean maps using ehtim library.

        """
        self.map = mapFile
        self.head = None
        self.comp = None
        self.cmap = None
        self.cmaph = dict()
        self.modFile = None
        self.modh = None
        if ccomp:
            self.comp,self.head,self.cmap = read_fits(self.map)
        else:
            with fits.open(mapFile) as hdulist:
                self.head = hdulist[0].header
                if load_eht:
                    import ehtim as eh
                    img = eh.image.load_fits(mapFile,aipscc=False)
                    self.cmap = img.imarr(pol='I')
                    maps_beam=[self.head['BMAJ']*np.pi/180,self.head['BMIN']*np.pi/180,self.head['BPA']*np.pi/180]
                    maps_ps = img.psize
                    ppb=PXPERBEAM(maps_beam[0],maps_beam[1],maps_ps)

                else:
                    img = hdulist[0].data
                    if img.ndim == 4:
                        self.cmap = img.reshape(img.shape[2],img.shape[3])
                    else:
                        self.cmap = img

        self.cmaph['noise'] = self.head['NOISE']
        self.cmaph['px_inc']= self.head['CDELT2']
        self.cmaph['naxis'] = self.head['NAXIS1']
        self.cmaph['beam']  = [self.head['BMAJ']*3.6e6,self.head['BMIN']*3.6e6,self.head['BPA']]
        self.cmaph['freq']  = np.around(self.head['crval3']*1e-9,1)
        self.cmaph['ppb']   = [PXPERBEAM(self.cmaph['beam'][0]*np.pi/180,self.cmaph['beam'][1]*np.pi/180,self.cmaph['px_inc']*3.6e6*np.pi/180)]
        self.cmaph['fov']   = self.cmaph['px_inc']*self.cmaph['naxis']*3.6e6

    def modelFile(self, modelFile):
        """A function to open model component fits files and get model fit parameters.

        Args:
            modelFile: the model component map fits file.

        Returns:
            None
        """
        self.modh = dict()
        with fits.open(modelFile) as f: #load model component parameters from fits file
            self.modFile = f[1].data
        self.modh['x']          = self.modFile['DELTAX']*3.6e6
        self.modh['y']          = self.modFile['DELTAY']*3.6e6
        self.modh['maj']        = self.modFile['MAJOR AX']*3.6e6
        self.modh['min']        = self.modFile['MINOR AX']*3.6e6
        self.modh['posa']       = self.modFile['POSANGLE']

    def plotMap(
        self,
        sigma: int = 3,
        fig_size: str ='screen',
        ra: int | bool = False,
        dec: int | bool = False,
        saveFile: str | bool = False,
        plot_mod: str | bool = False,
        plot_cntr: str = 'black',
        plot_color: bool = False,
        negative_contour: bool = True,
        cntr_color: str = 'black',
        model_color: str = 'red',
        cntr_lw: int | bool = False,
        sourcename: str | bool =False
    ):
        """Function to plot the clean map.
        Args:
            sigma: Sigma to use for the contour level definition.
            fig_size: String to define output medium for figure used in plotSet.py.
                Options are 'screen','aanda' (one column wide),'aanda*'
                (two columns wide),'beamer'
            ra: Provide th RA for the map.
            dec: Provide the DEC for the map.
            saveFile: File name for saving.
            plot_mod: False no model components are plotted. Set to model component
                file name if model components should be plotted on top of map.
            plot_cntr: True if contours should be plotted.
            plot_color: True for color plot
            cntr_color: custom contour color
            model_color: custom model component color
            cntr_lw: custom line width of contours
            sourcename: custom source name, if not set source is read from header.

        Returns:
            A PDF file containing the plot of the map.

        Raises:
            IOError: An error occured plotting the map.

        """
        font_color = 'black'
        if plot_color:
            box_color = 'white'
            box_alpha = 0.8
        else:
            box_color = 'black'
            box_alpha = 0.2
        if cntr_color=='black':
            if plot_mod:
                cntr_color = 'grey'
            if plot_color:
                cntr_color = 'white'
            else:
                cntr_color = 'black'
        if not cntr_lw:
            cntr_lw=1
        ####################
        # setting all parameters for plotting a clean image
        #####################
        if not ra:
            ra  = self.cmaph['fov']/3.
            dec = self.cmaph['fov']/5.
        if type(ra)==list:
            Dra = ra
            Ddec = dec
        else:
            Dra = [-ra,ra]
            Ddec = [-dec,dec]

        scale   = -self.cmaph['px_inc']*3.6e6
        sigma   = sigma
        level0= self.cmaph['noise']*sigma
        if negative_contour:
            sys.stdout.write("Plot also 1st negative contour.\n")
            lev = [-level0]
        else:
            lev=[]
        for i in range(0,10):
            lev.append(level0*2**i)
        xx = np.linspace(-self.cmaph['naxis']*0.5*scale, (self.cmaph['naxis']*0.5-1)*scale, self.cmaph['naxis'])
        yy = np.linspace(self.cmaph['naxis']*0.5*scale, -(self.cmaph['naxis']*0.5-1)*scale, self.cmaph['naxis'])
        vmax = 0.5*ma.amax(self.cmap)
        norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=vmax*1e3,base=10)


        ##################
        # Plotting
        ###################
        f,ax = plt.subplots()
        ax.axis('scaled')
        ax.set_xlim(Dra)
        ax.set_ylim(Ddec)
        ax.invert_xaxis()
        if sourcename:
            Source = sourcename
        else:
            Source = self.head['OBJECT']
        plot_title='{} - {} -{}GHz'.format(Source,self.head['DATE-OBS'],self.cmaph['freq'])
        bbox_props= dict(boxstyle='round', alpha=box_alpha, color=box_color)
        ax.annotate(plot_title, xy=(0.1,0.9), xycoords='axes fraction', fontsize=12, color=font_color, bbox=bbox_props, zorder=20)
        plotBeam(self.cmaph['beam'][0],self.cmaph['beam'][1],self.cmaph['beam'][2],ra,-dec+0.2,ax)
        if plot_cntr:
            cntr=ax.contour(xx,yy,self.cmap,linewidths=cntr_lw,levels=lev,colors=cntr_color,alpha=1)
        if plot_color:
            extent = np.max(xx),np.min(xx),np.min(yy),np.max(yy)
            im = ax.imshow(self.cmap*1e3,cmap=colormap,extent=extent,origin='lower', interpolation='gaussian')
            im.set_norm(norm)
            cbar = f.colorbar(im, ax=ax, location='top',pad=0.02,shrink=0.5)
            cbar.set_label(r'$S_\nu$ [mJy/beam]')

            plt.style.use('dark_background')

        #######
        # plot model if wanted 
        #######
        if plot_mod:
            if not type(self.modh) == dict:
                self.modelFile(plot_mod)
            Mx = self.modh['x']
            My = self.modh['y']
            Mposa = self.modh['posa']
            Mmaj    = self.modh['maj']
            Mmin    = self.modh['min']
            for j,xx in enumerate(Mx):
                e_comp = Ellipse([Mx[j],My[j]],Mmaj[j],Mmin[j],-Mposa[j]+90, color=model_color, zorder=2, fill=False,lw=0.5)
                ax.add_artist(e_comp)
                maj1_x = Mx[j]-np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                maj1_y = My[j]+np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                maj2_x = Mx[j]+np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                maj2_y = My[j]-np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5

                min1_x = Mx[j]-np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                min1_y = My[j]+np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                min2_x = Mx[j]+np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                min2_y = My[j]-np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                if maj1_y==maj2_y:
                    ax.plot(maj1_x,maj1_y, color = model_color,marker='+', markersize=5,lw=1)
                else:
                    ax.plot([maj1_x,maj2_x],[maj1_y,maj2_y], color = model_color, lw = 1)
                    ax.plot([min1_x,min2_x],[min1_y,min2_y], color = model_color, lw = 1)

        # set axis, labels, etc.
        ax.set(xlabel='RA [mas]', ylabel='DEC [mas]')
        ax.minorticks_on()
        if ra<=1 or dec<=1:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if saveFile:
            if type(saveFile) == str:
                saveFile_name = saveFile
            else:
                saveFile_name = self.map.split('.')[0]+'.pdf'
            fig_size='aanda'
            figsize=set_size(fig_size)
            #set_corrected_size(f,figsize)
            plt.savefig(saveFile_name,bbox_inches='tight')
            sys.stdout.write('Plot file {} saved.\n'.format(saveFile_name))
        else:
            plt.show()
        plt.clf()
        return f,ax
