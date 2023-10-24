#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import ehtim as eh
import VLBIana.modules.fit_functions as ff
from VLBIana.modules.plot_functions import *
from VLBIana.modules.jet_calculus import *

plt.style.use('pubstyle')
plt.ioff()
class CleanMap(object):
    '''
    Useage example:
    Map = CleanMap('maps/NGC1052_C.fits')
    Map.plotMap(plot_mod='models/NGC1052_C_model.fits')
    '''
    def __init__(self,mapFile, ccomp = False, load_eht = False):
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
        self.modh = dict()
        with fits.open(modelFile) as f:
            self.modFile = f[1].data
        self.modh['x']          = self.modFile['DELTAX']*3.6e6
        self.modh['y']          = self.modFile['DELTAY']*3.6e6
        self.modh['maj']        = self.modFile['MAJOR AX']*3.6e6
        self.modh['min']        = self.modFile['MINOR AX']*3.6e6
        self.modh['posa']       = self.modFile['POSANGLE']

    def plotMap(self,sigma=3,fig_size='screen',ra=False,dec=False,saveFile=False,plot_mod=False,plot_cntr=True,plot_color=False,cntr_color=False,model_color=False,cntr_lw=False,sourcename=False):
        font_color = 'black'
        if plot_color:
            box_color = 'white'
            box_alpha = 0.8
        else:
            box_color = 'black'
            box_alpha = 0.2
        if not cntr_color:
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
            Ddec    = [-dec,dec]

        scale   = -self.cmaph['px_inc']*3.6e6
        sigma   = sigma
        level0= self.cmaph['noise']*sigma
        lev=[]
        for i in range(0,10):
            lev.append(level0*2**i)
        xx  = np.linspace(-self.cmaph['naxis']*0.5*scale,(self.cmaph['naxis']*0.5-1)*scale,self.cmaph['naxis'])
        yy  = np.linspace(self.cmaph['naxis']*0.5*scale,-(self.cmaph['naxis']*0.5-1)*scale,self.cmaph['naxis'])
        vmax=0.5*ma.amax(self.cmap)
        norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=vmax*1e3,base=10)
        #norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)    im1 = ax[0,0].imshow(file1_plt*1e3,cmap=colormap,norm=norm,extent=extent1,zorder=1)


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
            cbar = f.colorbar(im, ax=ax, location='top',pad=0.02,shrink=0.5)#,ticks=[1e-3,1e-2,1e-1,1])
    #   cbar.ax.set_xticklabels([])
            cbar.set_label(r'$S_\nu$ [mJy/beam]')

    #        cax = divider.append_axes('right', size='5%', pad=0.05)
    #        cbar = f.colorbar(im, use_gridspec=True,cax=cax)
    #        cbar.set_label(r'Jy/beam')
            plt.style.use('dark_background')

        #######
        # plot model if wanted 
        #######
        if plot_mod:
            if not model_color:
                model_color='red'
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
