#!/usr/bin/env python
#VCAT/align/align_imagesEHTim.py

"""Provides the functions to aligne to maps using 2D-crosscorelation.

This script allows the user to align two maps using 2D-crosscorrelation
using the Python module scikit-image.

Examples:
    >>> from VCAT.align.aligm_imagesEHTim_final import *
    >>>
    >>>
"""

import string,math,sys,fileinput,glob,os,time
from scipy import *
import numpy as np
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import AxesGrid,make_axes_locatable
import matplotlib.pyplot as plt
from pylab import *
import os
from astropy.io import fits

##from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from matplotlib.patches import Circle,Ellipse
from skimage.draw import circle_perimeter,ellipse_perimeter
from vcat.VLBI_map_analysis.modules.plot_functions import *
import ehtim as eh
from vcat.helpers import get_common_beam

from vcat.VLBI_map_analysis.modules.jet_calculus import *

workDir     = os.getcwd()+'/'
plotDir = workDir+'plots/'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

def apply_shift(img,shift):
    """A function to apply a shift to an image.
    Args:
        img: the image
        shift: the shifts in y, and x-direction

    Returns:
        img
    """
    input_ = np.fft.fft2(img) #before it was np.fft.fftn(img)
    offset_image = fourier_shift(input_, shift=shift)
    imgalign = np.fft.ifft2(offset_image) #again before ifftn
    img2 = imgalign.real

    return img2

#=> migrated directly to ImageData now!

def align(img1,img2,inc1,inc2,mask1=False,mask2=False):
    '''Align two images using 2D-crosscorelation.

    Args:
        img1: image 1 to align
        img2: image 2 to align
        inc1: pixel increment of img1
        inc2: pixel increment of img2
        mask1: mask image 1
        mask2: mask image 2

    Returns:
    '''
    if mask1 is False:
        shift,error,diffphase = phase_cross_correlation((img1),(img2),upsample_factor=100)
    else:
        shift, _, _ = phase_cross_correlation((img1),(img2),upsample_factor=100,reference_mask=mask1,moving_mask=mask2)

    if mask1 is False:
        print ('register images new shift (y,x): [{} : {}] mas'.format(-shift[0]*inc1,-shift[1]*inc2))
        print ('register images new shift (y,x): {} px +- {}'.format(-shift,error))

    else:
        print ('register images new shift (y,x): [{} : {}] mas'.format(-shift[0]*inc1,-shift[1]*inc2))
        print ('register images new shift (y,x): {} px'.format(-shift))


    #shift img2 to found position
    img2 = apply_shift(img2,shift)

    #return aligned image and the computed shift
    if mask1 is False:
        return {'align':img2, 'shift':shift, 'error':error, 'diffphase': diffphase}
    else:
        return {'align':img2, 'shift':shift}


def masking(file1rb,file2rb,naxis=False,mask='ellipse',args=False,**kwargs):
    '''Define the masks that can be used for masking the images.

    Args:
        file1: Image1
        file2: Image2
        mask: 'npix_x','cut_left','cut_right','radius','ellipse','flux_cut'
        args: the arguments for the mask
            'npix_x': args=[npix_x,npixy]
            'cut_left': args = cut_left
            'cut_right': args = cut_right
            'radius': args = radius
            'ellipse': args = [e_maj,e_min,e_pa]
            'flux_cut: args = flux cut

    Returns:
        masks for both images
    '''
    mask1 = np.zeros_like(file1rb, dtype=bool)
    mask2 = np.zeros_like(file2rb, dtype=bool)
    if not naxis:
        naxis = len(file1rb)
    # cut out inner, optically thick part of the image
    if mask=='npix_x':
        npix_x = args[0] 
        npix_y = args[1]
        px_min_x = int(naxis/2-npix_x)
        px_max_x = int(naxis/2+npix_x)
        px_min_y = int(naxis/2-npix_y)
        px_max_y = int(naxis/2+npix_y)

        px_range_x = np.arange(px_min_x,px_max_x+1,1)
        px_range_y = np.arange(px_min_y,px_max_y+1,1)

        index=np.meshgrid(px_range_y,px_range_x)
        mask1[tuple(index)] = True
        mask2[tuple(index)] = True

    if mask=='cut_left':
        cut_left = args
        px_max = int(naxis/2.+cut_left)
        px_range_x = np.arange(0,px_max,1)
        mask1
        mask1[:,px_range_x] = True
        mask2[:,px_range_x] = True

    if mask=='cut_right':
        cut_right = args
        px_max = int(naxis/2-cut_right)
        px_range_x = np.arange(px_max,naxis,1)
        mask1[:,px_range_x] = True
        mask2[:,px_range_x] = True

    if mask=='radius':
        radius = args
        rr,cc = circle(int(len(file1rb)/2),int(len(file1rb)/2),radius)
        mask1[rr,cc] = True
        mask2[rr,cc] = True

    if mask=='ellipse':
        e_maj = args['e_args'][0]
        e_min = args['e_args'][1]
        e_pa  = args['e_args'][2]
        e_xoffset = args['e_xoffset']
        #e_xoffset = kwargs.get('e_xoffset',False)
        if e_xoffset!=False:
            x,y = int(len(file1rb)/2)+e_xoffset,int(len(file1rb)/2)
        else:
            x,y = int(len(file1rb)/2),int(len(file1rb)/2)
        if e_pa==False:
            e_pa = 0
        else:
            e_pa = e_pa
#        else:
#            #e_pa = self.maps_beam[0][2]
#            e_pa = 0
        rr,cc =ellipse(y,x,e_maj,e_min,rotation=e_pa*np.pi/180)
        mask1[rr,cc] = True
        mask2[rr,cc] = True

    if mask=='flux_cut':
        flux_cut = args
        mask1[file1>flux_cut*ma.amax(file1rb)] = True
        mask2[file2>flux_cut*ma.amax(file2rb)] = True

    return  (mask1,mask2)



class AlignMaps(object):
    '''Align VLBI maps using 2D cross-correlation.

    This class loads a set of images and alignes them.
    Optional one can mask the images to exclude optically thick
    regions.

    Attributes:
        maps: The input maps.
        masked_shift: using mask or not.
        beam = which beam to use? 'max','min','common','median'
        fig_size = output figsize, 'aanda'.
        plot_shifted = should the shifted map be plotted?
        plot_spix = should the spix map be plotted?
        plot_convolved = should the convolved maps be plotted?
        asize = figsize.
        sigma = what sigma to use for plotting.

    Examples:
        >>> aligned = AlignMaps()
        >>> aligned.align()
    '''
    def __init__(self,
                 maps,
                 masked_shift = True,
                 mask = False,
                 mask_args = False,
                 beam_arg = 'common',
                 fig_size = 'aanda',
                 plot_shifted = True,
                 plot_spix = True,
                 plot_convolved = True,
                 asize = 6,
                 sigma = 3):

#====================================================================
# define parameter

        self.files = [eh.image.load_fits(m,aipscc=True) for m in maps]
        self.fovx = np.array([m.fovx() for m in self.files])
        self.fovy = np.array([m.fovy() for m in self.files])
        self.header = [read_header(m) for m in maps]
        self.freq1 = np.round(self.header[0]['CRVAL3']*1e-9,1)
        self.freq2 = np.round(self.header[1]['CRVAL3']*1e-9,1)
        self.mask = mask
        self.sigma = sigma

        #before loading the beam, check if information is provided in header.
        try:
            self.maps_beam = [[h['BMAJ']*np.pi/180,h['BMIN']*np.pi/180,h['BPA']*np.pi/180] for h in self.header]
        except KeyError:
            sys.stdout.write('Beam information could not be extracted from header.\n')
            inp_bmaj = input('Provide bmaj: ')
            inp_bmin = input('Provide bmin: ')
            inp_bpa = input('Provide bpa: ')
            self.maps_beam = [float(inp_bmaj), float(inp_bmin), float(inp_bpa)]

        self.maps_ps = np.array([dm.psize for dm in self.files])
        self.naxis1 = (self.fovx/self.maps_ps).astype(int)
        self.naxis2 = (self.fovy/self.maps_ps).astype(int)
        ppb = [PXPERBEAM(bp[0],bp[1],pxi) for bp,pxi in zip(self.maps_beam,self.maps_ps)]
        r2m = 180/np.pi*3.6e6
        self.r2m = r2m

        #If no header info is available, estimate RMS noise from image
        '''Note: the 'noise' keyword in the header gives the theoretically
        expected rms noise based on the visibility data (difmap computes this in
        some way...). The real rms will usually be a factor of a few higher,
        depending on data quality. As replacement, the script now provisionally
        computes the rms in the bottom-left corner of the image, in a square of a
        tenth of the image size. Use with care, this should be replaced with a more
        meaningful estimate in the future.
        '''
        try:
            noise1=self.header[0]['NOISE']
        except KeyError:
            I1 = fits.getdata(maps[0])
            noise1=1.8*np.std(I1[0,0,0:round(self.naxis1[0]/10.),0:round(self.naxis2[0]/10.)])
        sys.stdout.write('RMS noise map 1: {:.5f} Jy/beam\n'.format(noise1))
        try:
            noise2=self.header[1]['NOISE']
        except KeyError:
            from astropy.io import fits
            I2 = fits.getdata(maps[1])
            noise2=1.8*np.std(I2[0,0,0:round(self.naxis1[1]/10.),0:round(self.naxis2[1]/10.)])
        sys.stdout.write('RMS noise map 2: {:.5f} Jy/beam\n'.format(noise2))

        # derive the common beam to be used
        if type(beam_arg) == list:
            _maj,_min,_pos = arg
            _maj /= r2m
            _min /= r2m
            _pos = _pos*180/np.pi
            self.common_beam = [_maj,_min,_pos]
        else:
            self.common_beam = get_common_beam([self.maps_beam[0][0],self.maps_beam[1][0]],[self.maps_beam[0][1],self.maps_beam[1][1]],[self.maps_beam[0][2],self.maps_beam[1][2]],arg=beam_arg)
        sys.stdout.write('Will use common beam of ({} , {} ) mas at PA {} degree.\n'.format(self.common_beam[0]*r2m,self.common_beam[1]*r2m,self.common_beam[2]*180/np.pi))#/np.pi))

        #derive common parameter for the common map

        self.common_ps = self.maps_ps.min()
        self.common_fov = min([min(x,y) for x,y in zip(self.fovx,self.fovy)])
        self.common_naxis = int(self.common_fov/self.common_ps)
        self.common_ppb = PXPERBEAM(self.common_beam[0],self.common_beam[1],self.common_ps)
        self.common_ppb = PXPERBEAM(self.common_beam[0],self.common_beam[1],self.common_ps)

        self.noise1_r = noise1/ppb[0]*self.common_ppb
        self.noise2_r = noise2/ppb[1]*self.common_ppb

        self.noise1 = noise1
        self.noise2 = noise2
    # regrid and blur the clean maps
    # I actually have no idea why this logic was checked... need to test and find out
    # check for same fov and pixel size
    #if np.logical_and(self.maps_ps[0] != self.maps_ps[1], self.fovx[0]!=self.fovy[1]):

        self.file1regrid = self.files[0].regrid_image(self.common_fov, self.common_naxis, interp='linear')
        self.file1regridblur = self.file1regrid.blur_gauss(self.common_beam,frac=1)
        self.file2regrid = self.files[1].regrid_image(self.common_fov, self.common_naxis, interp='linear')
        self.file2regridblur = self.file2regrid.blur_gauss(self.common_beam, frac=1)
        self.file1rb = self.file1regridblur.imarr(pol='I').copy()
        self.file2rb = self.file2regridblur.imarr(pol='I').copy()

        if self.mask: #use a mask during alignment
            if self.mask == "ellipse":
                if mask_args["e_args"][2] == "beam":
                    mask_args["e_args"][2] = self.common_beam[2]*180/np.pi
            mask1,mask2 = masking(self.file1rb,self.file2rb,self.common_naxis,mask=self.mask,args=mask_args)
            self.file1rbm = self.file1rb.copy()
            self.file2rbm = self.file2rb.copy()
            self.file1rbm[mask1] = 0
            self.file2rbm[mask2] = 0
            self.file1rbm_plt = self.file1rbm*self.common_ppb
            self.file2rbm_plt = self.file2rbm*self.common_ppb


        # align image, using a mask
        # ps covnerted from radperpx to masperpx
            if masked_shift: #use the mask during the cross-correlation
                sys.stdout.write('Will derive the shift using the mask during cross-correlation\n')
                self.file2rb_shift = align(self.file1rb,self.file2rb,self.common_ps*180/np.pi*3.6e6,self.common_ps*180/np.pi*3.6e6,mask1=~mask1,mask2=~mask2)
            else: #use already masked images
                sys.stdout.write('Will derive the shift using already masked images\n')
                self.file2rb_shift = align(self.file1rbm,self.file2rbm,self.common_ps*180/np.pi*3.6e6,self.common_ps*180/np.pi*3.6e6)
        else: #without any mask
            sys.stdout.write('Will derive the shift using already unmasked images\n')
            self.file2rb_shift = align(self.file1rb,self.file2rb,self.common_ps*180/np.pi*3.6e6,self.common_ps*180/np.pi*3.6e6)

        #define arrays of images for plotting
        self.file1_plt = self.files[0].regrid_image(self.fovx[0],self.naxis1[0]).blur_gauss(self.maps_beam[0],frac=1).imarr(pol='I')
        self.file2_plt = self.files[1].regrid_image(self.fovx[1],self.naxis1[1]).blur_gauss(self.maps_beam[1],frac=1).imarr(pol='I')
        self.file1_plt *= ppb[0]
        self.file2_plt *= ppb[1]
        self.file1rb_plt = self.file1rb*self.common_ppb
        self.file2rb_plt = self.file2rb*self.common_ppb
        self.file2rb_shift_plt = apply_shift(self.file2rb,self.file2rb_shift['shift'])* self.common_ppb


    def plot_spix(self):
        f,ax = plt.subplots()
        #if fig_size=='aanda*':
        fig_size='aanda'

        sigma = self.sigma

        ra=(self.common_fov/eh.RADPERUAS/1e3/2)-1
        #dec=common_fov/eh.RADPERUAS/1e3/3
        dec = ra*7/10
        ra_min=-ra
        ra_max=ra
        dec_min=-dec
        dec_max=dec
        scale1  = self.maps_ps[0]*180/np.pi*3.6e6
        scale2  = self.maps_ps[1]*180/np.pi*3.6e6
        common_scale = self.common_ps*180/np.pi*3.6e6

        x1=np.linspace(-self.naxis1[0]*0.5*scale1,(self.naxis1[0]*0.5-1)*scale1,self.naxis1[0])
        x2=np.linspace(self.naxis1[1]*0.5*scale2,-(self.naxis1[1]*0.5-1)*scale2,self.naxis1[1])
        xc=np.linspace(self.common_naxis*0.5*common_scale,-(self.common_naxis*0.5-1)*common_scale,self.common_naxis)
        y1=np.linspace(-self.naxis2[0]*0.5*scale1,(self.naxis2[0]*0.5-1)*scale1,self.naxis2[0])
        y2=np.linspace(self.naxis2[1]*0.5*scale2,-(self.naxis2[1]*0.5-1)*scale2,self.naxis2[1])
    #    yc=np.linspace(common_naxis*0.5*common_scale,-(common_naxis*0.5-1)*common_scale,common_naxis)
        yc = xc
        extent1 = np.max(x1), np.min(x1), np.min(y1), np.max(y1)
        extent2 = np.max(x2), np.min(x2), np.min(y2), np.max(y2)
        extentc = np.max(xc), np.min(xc), np.min(yc), np.max(yc)

        level0  = min([self.noise1,self.noise2,self.noise1_r,self.noise2_r])*sigma
        lev=[]
        for i in range(0,10):
            lev.append(level0*2**i)
        level1r = self.noise1_r*sigma
        lev1_r=[]
        for i in range(0,10):
            lev1_r.append(level1r*2**i)
        level2r = self.noise2_r*sigma
        lev2_r=[]
        for i in range(0,10):
            lev2_r.append(level2r*2**i)

        axe_ratio='scaled'

        #####################
        file1rb_plt = np.flipud(self.file1rb_plt)
        file2rb_shift_plt = np.flipud(self.file2rb_shift_plt)
        spix1 = file1rb_plt*(file1rb_plt > self.noise1*sigma) #replaces indices where condition is not met with 0
        spix2 = file2rb_shift_plt*(file2rb_shift_plt > self.noise2*sigma)
        spix1[spix1==0] = self.noise1*sigma
        spix2[spix2==0] = self.noise2*sigma
        a = np.log10(spix2/spix1)/np.log10(self.freq2/self.freq1)

        spix_vmin,spix_vmax=-3,5
        sys.stdout.write('\nSpectral index max(alpha)={} - min(alpha)={}\nCutoff {}<alpha<{}\n'.format(ma.amax(a),ma.amin(a),spix_vmin,spix_vmax))
        a[a<spix_vmin] = spix_vmin
        a[a>spix_vmax] = spix_vmax
        a[spix2==self.noise2*sigma] = spix_vmin

#        level10 = self.noise1*sigma
#        lev1=[]
#        level20 = self.noise2*sigma
#        lev2=[]
#
#        for i in range(0,10):
#            lev1.append(level10*2**i)
#            lev2.append(level20*2**i)

        cset = ax.contour(spix1,linewidths=[0.5],levels=lev1_r,colors=['grey'], extent=extent2,origin='lower',alpha=0.7)
        im = ax.imshow(a,cmap='hot_r',origin='lower',extent= extent2,vmin=spix_vmin,vmax=spix_vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = f.colorbar(im, use_gridspec=True,cax=cax)
        cbar.set_label(r'$\alpha$')
        h1,_ = cset.legend_elements()
        ax.legend([h1[0]],['{} GHz'.format(self.freq1)],loc=2,prop={'size':asize+1})

        ax.axis('scaled')
        ax.set_xlabel('RA [mas]')
        ax.set_ylabel('Relative Dec [mas]')
        ax.set_xlim(ra_max,ra_min)
        ax.set_ylim(dec_min,dec_max)
        ax.minorticks_on()

        figsize=set_size(fig_size)
        set_corrected_size(f,figsize)
        plotname =plotDir+'spectral_index_between_{:d}_{:d}.pdf'.format(int(self.freq1),int(self.freq2))
        plt.savefig(plotname,bbox_inches='tight')
        sys.stdout.write('Save spixmap to {}\n'.format(plotname))

    def plot_shifted_maps(self,plot_convolved=True,plot_shifted=True,colormap='hot',**kwargs):

        fig_size ='aanda'
        file2rb_shift_plt = self.file2rb_shift_plt
        sigma = self.sigma
        ###

        ra=(self.common_fov/eh.RADPERUAS/1e3/2)-1
        #dec=common_fov/eh.RADPERUAS/1e3/3
        dec = ra*7/10
        ra_min=-ra
        ra_max=ra
        dec_min=-dec
        dec_max=dec
        scale1  = self.maps_ps[0]*180/np.pi*3.6e6
        scale2  = self.maps_ps[1]*180/np.pi*3.6e6
        common_scale = self.common_ps*180/np.pi*3.6e6
        common_beam = self.common_beam

        x1=np.linspace(-self.naxis1[0]*0.5*scale1,(self.naxis1[0]*0.5-1)*scale1,self.naxis1[0])
        x2=np.linspace(self.naxis1[1]*0.5*scale2,-(self.naxis1[1]*0.5-1)*scale2,self.naxis1[1])
        xc=np.linspace(self.common_naxis*0.5*common_scale,-(self.common_naxis*0.5-1)*common_scale,self.common_naxis)
        y1=np.linspace(-self.naxis2[0]*0.5*scale1,(self.naxis2[0]*0.5-1)*scale1,self.naxis2[0])
        y2=np.linspace(self.naxis2[1]*0.5*scale2,-(self.naxis2[1]*0.5-1)*scale2,self.naxis2[1])
    #    yc=np.linspace(common_naxis*0.5*common_scale,-(common_naxis*0.5-1)*common_scale,common_naxis)
        yc = xc
        extent1 = np.max(x1), np.min(x1), np.min(y1), np.max(y1)
        extent2 = np.max(x2), np.min(x2), np.min(y2), np.max(y2)
        extentc = np.max(xc), np.min(xc), np.min(yc), np.max(yc)

        level0  = min([self.noise1,self.noise2,self.noise1_r,self.noise2_r])*sigma
        lev=[]
        for i in range(0,10):
            lev.append(level0*2**i)

        level1r = self.noise1_r*sigma
        lev1_r=[]
        for i in range(0,10):
            lev1_r.append(level1r*2**i)
        level2r = self.noise2_r*sigma
        lev2_r=[]
        for i in range(0,10):
            lev2_r.append(level2r*2**i)

        file1_plt = self.file1_plt
        file2_plt = self.file2_plt
        file1rb_plt = self.file1rb_plt
        file2rb_plt = self.file2rb_plt
        if self.mask != False:
            file1rbm_plt = self.file1rbm_plt
            file2rbm_plt = self.file2rbm_plt

        axe_ratio='scaled'
        r2m = self.r2m
    ################# Plot 1 ##########################
        if plot_convolved:
            fig_size=('aanda')
            f = plt.figure(constrained_layout=True)
            gs = f.add_gridspec(2, 2, hspace=0, wspace=0)
            ax = gs.subplots(sharex='col', sharey='row')

            #f,ax = plt.subplots(2,2,sharex='col',sharey='row',gridspec_kw={'hspace':0,'wspace':0})

            l1 = '{} GHz original'.format(self.freq1)
            l2 = '{} GHz original'.format(self.freq2)
            l3 = '{} GHz regrid $+$ blur'.format(self.freq1)
            l4 = '{} GHz regrid $+$ blur'.format(self.freq2)
            label=[l1,l2,l3,l4]

            imax = max([ma.amax(ii) for ii in [file1_plt,file2_plt,file1rb_plt,file2rb_plt]])
            norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)
            im1 = ax[0,0].imshow(file1_plt*1e3,cmap=colormap,norm=norm,extent=extent1,zorder=1)
            plotBeam(self.maps_beam[0][0]*r2m,self.maps_beam[0][1]*r2m,self.maps_beam[0][2]*180/np.pi,ra,-dec,ax=ax[0,0])
            im2 = ax[0,1].imshow(file2_plt*1e3,cmap=colormap,norm=norm,extent=extent2,zorder=1)
            plotBeam(self.maps_beam[1][0]*r2m,self.maps_beam[1][1]*r2m,self.maps_beam[1][2]*180/np.pi,ra,-dec,ax=ax[0,1])
            im3 = ax[1,0].imshow(file1rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
            plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax=ax[1,0])
            im4 = ax[1,1].imshow(file2rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
            plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax=ax[1,1])

            cbar = f.colorbar(im1, ax=ax[:,:], location='top',pad=0.01,shrink=0.9,aspect=35)#,ticks=[1e-3,1e-2,1e-1,1])
        #   cbar.ax.set_xticklabels([])
            cbar.set_label(r'$S_\nu$ [mJy/beam]')

            for aa,ll in zip(ax.flat,label):
                aa.xaxis.set_minor_locator(AutoMinorLocator())
                aa.yaxis.set_minor_locator(AutoMinorLocator())
                aa.axis(axe_ratio)
                aa.set_xlim(ra_max,ra_min)
                aa.set_ylim(dec_min,dec_max)
                aa.annotate(ll, xy=(0.1,0.9),xycoords='axes fraction',size=asize,color='w')
                aa.tick_params(direction='in',which='both',color='w')
            ax[0,0].set(ylabel='Relative Dec [mas]')
            ax[1,0].set(xlabel='RA [mas]',ylabel='Dec [mas]')
            ax[1,1].set(xlabel='RA [mas]')
            figsize=set_size(fig_size,subplots=(2,2),ratio=0.88)
            set_corrected_size(f,figsize)
            plotname = plotDir+'{:d}GHz_convolved_with_{:d}GHz.pdf'.format(int(self.freq2),int(self.freq1))
            f.savefig(plotname,bbox_inches='tight')
            sys.stdout.write('Save masked map plot to {}\n'.format(plotname))

    ######################################################  

    #   plt.cla()
        if plot_shifted:
            #fig_size=('aanda')
            f = plt.figure(constrained_layout=True)
            f = plt.figure()
            gs = f.add_gridspec(2, 2, hspace=0,wspace=0)
            ax = gs.subplots(sharex='col',sharey='row')

            l1 = '{} GHz regrid $+$ blur'.format(self.freq1)
            l2 = '{} GHz regrid $+$ blur'.format(self.freq2)
            l3 = '{0} GHz/ {1} GHz not shifted'.format(self.freq1,self.freq2)
            l4 = '{0} GHz/ {1} GHz shifted'.format(self.freq1,self.freq2)
            label=[l1,l2,l3,l4]

            norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)
            if self.mask == False:
                imax = max([ma.amax(ii) for ii in [file1rb_plt,file2rb_plt]])
                im1 = ax[0,0].imshow(file1rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
                im2 = ax[0,1].imshow(file2rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
            else:
                imax = max([ma.amax(ii) for ii in [file1rbm_plt,file2rbm_plt]])
                im1 = ax[0,0].imshow(file1rbm_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
                im2 = ax[0,1].imshow(file2rbm_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)

            plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax[0,0])
            plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax[0,1])

            cntr1=ax[1,0].contour(np.flipud(file1rb_plt),linewidths=0.5,levels=lev1_r,colors='grey',extent=extentc,alpha=1)
            cntr2=ax[1,0].contour(np.flipud(file2rb_plt),linewidths=0.5,levels=lev2_r,colors='darkblue',extent=extentc,alpha=0.6)
            h1,_ = cntr1.legend_elements()
            h2,_ = cntr2.legend_elements()
            ax[1,0].legend([h1[0],h2[0]],['{}GHz'.format(self.freq1),'{}GHz'.format(self.freq2)],loc=3,prop={'size':asize})
            #
            cntr1=ax[1,1].contour(np.flipud(file1rb_plt),linewidths=0.5,levels=lev1_r,colors='grey',extent=extentc)
            cntr2=ax[1,1].contour(np.flipud(file2rb_shift_plt),linewidths=0.5,levels=lev2_r,colors='darkblue',extent=extentc,alpha=0.6)
            h1,_ = cntr1.legend_elements()
            h2,_ = cntr2.legend_elements()
            ax[1,1].legend([h1[0],h2[0]],['{}GHz'.format(self.freq1),'{}GHz'.format(self.freq2)],loc=3,prop={'size':asize})

    ##      cax = divider.append_axes('top', size='5%', pad=0.01)
            cbar = f.colorbar(im1, ax=ax[:,:], location='top',pad=0.01,shrink=0.9,aspect=35)
            cbar.set_label(r'$S_\nu$ [mJy/beam]')

            for aa,ll in zip(ax.flat,label):
                aa.axis(axe_ratio)
                aa.xaxis.set_minor_locator(AutoMinorLocator())
                aa.yaxis.set_minor_locator(AutoMinorLocator())
                aa.set_xlim(ra_max,ra_min)
                aa.set_ylim(dec_min,dec_max)

            ax[0,0].annotate(l1, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1,color='w')
            ax[0,1].annotate(l2, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1,color='w')
            ax[1,0].annotate(l3, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1)
            ax[1,1].annotate(l4, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1)

            ax[0,0].tick_params(direction='in',which='both',color='w')
            ax[0,1].tick_params(direction='in',which='both',color='w')
            ax[1,1].tick_params(direction='in',which='both',axis='y')
            ax[0,0].set(ylabel='Dec [mas]')
            ax[1,0].set(xlabel='RA [mas]',ylabel='Dec [mas]')
            ax[1,1].set(xlabel='RA [mas]')

            figsize=set_size(fig_size,subplots=(2,2),ratio=0.88)
            set_corrected_size(f,figsize)
            plotname = plotDir+'shifted_maps_{:d}GHz_{:d}GHz.pdf'.format(int(self.freq1),int(self.freq2))
            f.savefig(plotname,bbox_inches='tight')
            sys.stdout.write('Save shifted map plot to {}\n'.format(plotname))

        plt.close('all')

def plot_spix(self):
################################


    ### Option to apply manual shift here! ###
    if 'shift_manual' in kwargs.keys():
        shift_manual = kwargs.get('shift_manual', False)
        print('Using manual shift of (y,x): [{} : {}] mas '.format(shift_manual[0], shift_manual[1]))
        file2rb_shift_plt = apply_shift(file2rb,np.round(shift_manual/(common_ps*180/np.pi*3.6e6),0))* common_ppb
        file2rb_shift['shift'] = np.round(shift_manual/(common_ps*180/np.pi*3.6e6),0)
    else:
        file2rb_shift_plt = apply_shift(file2rb,file2rb_shift['shift'])* common_ppb
    ###

    ra=(common_fov/eh.RADPERUAS/1e3/2)-1
    #dec=common_fov/eh.RADPERUAS/1e3/3
    dec = ra*7/10
    ra_min=-ra
    ra_max=ra
    dec_min=-dec
    dec_max=dec
    scale1  = self.maps_ps[0]*180/np.pi*3.6e6
    scale2  = self.maps_ps[1]*180/np.pi*3.6e6
    common_scale = common_ps*180/np.pi*3.6e6

    x1=np.linspace(-self.naxis1[0]*0.5*scale1,(self.naxis1[0]*0.5-1)*scale1,self.naxis1[0])
    x2=np.linspace(self.naxis1[1]*0.5*scale2,-(self.naxis1[1]*0.5-1)*scale2,self.naxis1[1])
    xc=np.linspace(common_naxis*0.5*common_scale,-(common_naxis*0.5-1)*common_scale,common_naxis)
    y1=np.linspace(-self.naxis2[0]*0.5*scale1,(self.naxis2[0]*0.5-1)*scale1,self.naxis2[0])
    y2=np.linspace(self.naxis2[1]*0.5*scale2,-(self.naxis2[1]*0.5-1)*scale2,self.naxis2[1])
#    yc=np.linspace(common_naxis*0.5*common_scale,-(common_naxis*0.5-1)*common_scale,common_naxis)
    yc = xc
    extent1 = np.max(x1), np.min(x1), np.min(y1), np.max(y1)
    extent2 = np.max(x2), np.min(x2), np.min(y2), np.max(y2)
    extentc = np.max(xc), np.min(xc), np.min(yc), np.max(yc)

    level0  = min([noise1,noise2,noise1_r,noise2_r])*sigma
    lev=[]
    for i in range(0,10):
        lev.append(level0*2**i)

    level1r = noise1_r*sigma
    lev1_r=[]
    for i in range(0,10):
        lev1_r.append(level1r*2**i)
    level2r = noise2_r*sigma
    lev2_r=[]
    for i in range(0,10):
        lev2_r.append(level2r*2**i)

    axe_ratio='scaled'
################# Plot 1 ##########################
    if plot_convolved:
        #fig_size=('aanda')
        f = plt.figure(constrained_layout=True)
        gs = f.add_gridspec(2, 2, hspace=0, wspace=0)
        ax = gs.subplots(sharex='col', sharey='row')

        #f,ax = plt.subplots(2,2,sharex='col',sharey='row',gridspec_kw={'hspace':0,'wspace':0})

        l1 = '{} GHz original'.format(self.freq1)
        l2 = '{} GHz original'.format(self.freq2)
        l3 = '{} GHz regrid $+$ blur'.format(self.freq1)
        l4 = '{} GHz regrid $+$ blur'.format(self.freq2)
        label=[l1,l2,l3,l4]

        imax = max([ma.amax(ii) for ii in [file1_plt,file2_plt,file1rb_plt,file2rb_plt]])
        norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)
        im1 = ax[0,0].imshow(file1_plt*1e3,cmap=colormap,norm=norm,extent=extent1,zorder=1)
        plotBeam(self.maps_beam[0][0]*r2m,self.maps_beam[0][1]*r2m,self.maps_beam[0][2]*180/np.pi,ra,-dec,ax=ax[0,0])
        im2 = ax[0,1].imshow(file2_plt*1e3,cmap=colormap,norm=norm,extent=extent2,zorder=1)
        plotBeam(self.maps_beam[1][0]*r2m,self.maps_beam[1][1]*r2m,self.maps_beam[1][2]*180/np.pi,ra,-dec,ax=ax[0,1])
        im3 = ax[1,0].imshow(file1rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
        plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax=ax[1,0])
        im4 = ax[1,1].imshow(file2rb_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
        plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax=ax[1,1])

        cbar = f.colorbar(im1, ax=ax[:,:], location='top',pad=0.01,shrink=0.9,aspect=35)#,ticks=[1e-3,1e-2,1e-1,1])
    #   cbar.ax.set_xticklabels([])
        cbar.set_label(r'$S_\nu$ [mJy/beam]')

        for aa,ll in zip(ax.flat,label):
            aa.xaxis.set_minor_locator(AutoMinorLocator())
            aa.yaxis.set_minor_locator(AutoMinorLocator())
            aa.axis(axe_ratio)
            aa.set_xlim(ra_max,ra_min)
            aa.set_ylim(dec_min,dec_max)
            aa.annotate(ll, xy=(0.1,0.9),xycoords='axes fraction',size=asize,color='w')
            aa.tick_params(direction='in',which='both',color='w')
        ax[0,0].set(ylabel='Relative Dec [mas]')
        ax[1,0].set(xlabel='RA [mas]',ylabel='Dec [mas]')
        ax[1,1].set(xlabel='RA [mas]')
        figsize=set_size(fig_size,subplots=(2,2),ratio=0.88)
        set_corrected_size(f,figsize)
        f.savefig(plotDir+'{:d}GHz_convolved_with_{:d}GHz.pdf'.format(int(self.freq2),int(self.freq1)),bbox_inches='tight')

######################################################  

#   plt.cla()
    if plot_shifted:
        #fig_size=('aanda')
        f = plt.figure(constrained_layout=True)
        f = plt.figure()
        gs = f.add_gridspec(2, 2, hspace=0,wspace=0)
        ax = gs.subplots(sharex='col',sharey='row')

        l1 = '{} GHz regrid $+$ blur'.format(self.freq1)
        l2 = '{} GHz regrid $+$ blur'.format(self.freq2)
        l3 = '{0} GHz/ {1} GHz not shifted'.format(self.freq1,self.freq2)
        l4 = '{0} GHz/ {1} GHz shifted'.format(self.freq1,self.freq2)
        label=[l1,l2,l3,l4]

        imax = max([ma.amax(ii) for ii in [file1rbm_plt,file2rbm_plt]])
        norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)
        im1 = ax[0,0].imshow(file1rbm_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
        plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax[0,0])
        im2 = ax[0,1].imshow(file2rbm_plt*1e3,cmap=colormap,norm=norm,extent=extentc,zorder=1)
        plotBeam(common_beam[0]*r2m,common_beam[1]*r2m,common_beam[2]*180/np.pi,ra,-dec,ax[0,1])

        cntr1=ax[1,0].contour(np.flipud(file1rb_plt),linewidths=0.5,levels=lev1_r,colors='grey',extent=extentc,alpha=1)
        cntr2=ax[1,0].contour(np.flipud(file2rb_plt),linewidths=0.5,levels=lev2_r,colors='darkblue',extent=extentc,alpha=0.6)
        h1,_ = cntr1.legend_elements()
        h2,_ = cntr2.legend_elements()
        ax[1,0].legend([h1[0],h2[0]],['{}GHz'.format(self.freq1),'{}GHz'.format(self.freq2)],loc=3,prop={'size':asize})
        #
        cntr1=ax[1,1].contour(np.flipud(file1rb_plt),linewidths=0.5,levels=lev1_r,colors='grey',extent=extentc)
        cntr2=ax[1,1].contour(np.flipud(file2rb_shift_plt),linewidths=0.5,levels=lev2_r,colors='darkblue',extent=extentc,alpha=0.6)
        h1,_ = cntr1.legend_elements()
        h2,_ = cntr2.legend_elements()
        ax[1,1].legend([h1[0],h2[0]],['{}GHz'.format(self.freq1),'{}GHz'.format(self.freq2)],loc=3,prop={'size':asize})

##      cax = divider.append_axes('top', size='5%', pad=0.01)
        cbar = f.colorbar(im1, ax=ax[:,:], location='top',pad=0.01,shrink=0.9,aspect=35)
        cbar.set_label(r'$S_\nu$ [mJy/beam]')

        for aa,ll in zip(ax.flat,label):
            aa.axis(axe_ratio)
            aa.xaxis.set_minor_locator(AutoMinorLocator())
            aa.yaxis.set_minor_locator(AutoMinorLocator())
            aa.set_xlim(ra_max,ra_min)
            aa.set_ylim(dec_min,dec_max)

        ax[0,0].annotate(l1, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1,color='w')
        ax[0,1].annotate(l2, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1,color='w')
        ax[1,0].annotate(l3, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1)
        ax[1,1].annotate(l4, xy=(0.1,0.9),xycoords='axes fraction',size=asize+1)

        ax[0,0].tick_params(direction='in',which='both',color='w')
        ax[0,1].tick_params(direction='in',which='both',color='w')
        ax[1,1].tick_params(direction='in',which='both',axis='y')
        ax[0,0].set(ylabel='Dec [mas]')
        ax[1,0].set(xlabel='RA [mas]',ylabel='Dec [mas]')
        ax[1,1].set(xlabel='RA [mas]')

        figsize=set_size(fig_size,subplots=(2,2),ratio=0.88)
        set_corrected_size(f,figsize)
        f.savefig(plotDir+'shifted_maps_{:d}GHz_{:d}GHz.pdf'.format(int(self.freq1),int(self.freq2)),bbox_inches='tight')
        plt.close('all')

#########
#########################
    # plot spix map
    if plot_spix:
        f,ax = plt.subplots()
        #if fig_size=='aanda*':
    #       fig_size='aanda'

        file1rb_plt = np.flipud(file1rb_plt)
        file2rb_shift_plt = np.flipud(file2rb_shift_plt)
        spix1 = file1rb_plt*(file1rb_plt > noise1*sigma) #replaces indices where condition is not met with 0
        spix2 = file2rb_shift_plt*(file2rb_shift_plt > noise2*sigma)
        spix1[spix1==0] = noise1*sigma
        spix2[spix2==0] = noise2*sigma
        a = np.log10(spix2/spix1)/np.log10(self.freq2/self.freq1)

        spix_vmin,spix_vmax=-3,5
        sys.stdout.write('\nSpectral index max(alpha)={} - min(alpha)={}\nCutoff {}<alpha<{}\n'.format(ma.amax(a),ma.amin(a),spix_vmin,spix_vmax))
        a[a<spix_vmin] = spix_vmin
        a[a>spix_vmax] = spix_vmax
        a[spix2==noise2*sigma] = spix_vmin

        level10 = noise1*sigma
        lev1=[]
        level20 = noise2*sigma
        lev2=[]

        for i in range(0,10):
            lev1.append(level10*2**i)
            lev2.append(level20*2**i)

        cset = ax.contour(spix1,linewidths=[0.5],levels=lev1_r,colors=['grey'], extent=extent2,origin='lower',alpha=0.7)
        im = ax.imshow(a,cmap='hot_r',origin='lower',extent= extent2,vmin=spix_vmin,vmax=spix_vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = f.colorbar(im, use_gridspec=True,cax=cax)
        cbar.set_label(r'$\alpha$')
        h1,_ = cset.legend_elements()
        ax.legend([h1[0]],['{} GHz'.format(self.freq1)],loc=2,prop={'size':asize+1})

        ax.axis('scaled')
        ax.set_xlabel('RA [mas]')
        ax.set_ylabel('Relative Dec [mas]')
        ax.set_xlim(ra_max,ra_min)
        ax.set_ylim(dec_min,dec_max)
        ax.minorticks_on()

        figsize=set_size(fig_size)
        set_corrected_size(f,figsize)

        plt.savefig(plotDir+'spectral_index_between_{:d}_{:d}.pdf'.format(int(self.freq1),int(self.freq2)),bbox_inches='tight')
#############################
    plt.close('all')

    shift_export=file2rb_shift['shift'].copy()
    sys.stdout.write('final shift: {}'.format(shift_export))
    shift_export[0]*=common_ps*180/np.pi*3.6e6
    shift_export[1]*=common_ps*180/np.pi*3.6e6
    if masked_shift:
        sys.stdout.write('shift in mas: {}'.format(shift_export))
    else:
        error_export=file2rb_shift['error'].copy()
        error_export*=common_ps*180/np.pi*3.6e6
        sys.stdout.write('shift in mas: {}\pm{}'.format(shift_export,error_export))

    if masked_shift:
        return {'file1':file1regridblur,'file2':file2regridblur,'shift':shift_export,'increment_dec':common_ps*3.6e6,'increment_ra':common_ps*3.6e6}
    else:
        return {'file1':file1regridblur,'file2':file2regridblur,'shift':shift_export,'increment_dec':common_ps*3.6e6,'increment_ra':common_ps*3.6e6,'error':error_export,'diffphase':file2rb_shift['diffphase']}
