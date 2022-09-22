#!/usr/bin/env python
import string
import math
import sys
import fileinput
import glob
import os
import time
from numpy import *
from scipy import *
import numpy as np
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import AxesGrid,make_axes_locatable
import matplotlib.pyplot as plt
from pylab import *
import os

##from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from matplotlib.patches import Circle,Ellipse
from skimage.draw import circle_perimeter,ellipse_perimeter
from VCAT.modules.plot_functions import *
import ehtim as eh

from VCAT.modules.jet_calculus import *

workDir     = os.getcwd()+'/'
plotDir = workDir+'plots/'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

def apply_shift(img,shift):
    offset_image = fourier_shift(np.fft.fftn(img), shift)
    imgalign = np.fft.ifftn(offset_image)
    img2 = imgalign.real

    return img2

def align(img1,img2,inc1,inc2,mask1=False,mask2=False):
    if mask1 is False:
        shift,error,diffphase = phase_cross_correlation((img1),(img2),upsample_factor=100)
    else:
        shift = phase_cross_correlation((img1),(img2),upsample_factor=100,reference_mask=mask1,moving_mask=mask2)

    if mask1 is False:
        print ('register images new shift (y,x): [{} : {}] mas'.format(shift[0]*inc1,shift[1]*inc2))
        print ('register images new shift (y,x): {} px +- {}'.format(shift,error))

    else:
        print ('register images new shift (y,x): [{} : {}] mas'.format(shift[0]*inc1,shift[1]*inc2))
        print ('register images new shift (y,x): {} px'.format(shift))


    #shift img2 to found position
    img2 = apply_shift(img2,shift)

    #return aligned image and the computed shift
    if mask1 is False:
        return {'align':img2, 'shift':shift, 'error':error, 'diffphase': diffphase}
    else:
        return {'align':img2, 'shift':shift}
#==============================================

###run the script
'''
How to run the scipt:

def plot_aligned_maps(maps,flux_cut=False,radius=False,npix_x=False,npix_y=False,e_maj=False,e_min=False,e_pa=False,masked_shift=False):

def plot_aligned_maps(maps,masked_shift=True, **kwargs):
    read in images
    files       = maps
'''
def plot_aligned_maps(maps,masked_shift=True, beam='max', fig_size='aanda', plot_shifted=True,plot_spix=True,plot_convolved=True, asize=6, **kwargs):
    '''Derive shifts and plot images
    All angles are in rad and converted if needed.
    If the beam is given explicitly, please write it in terms of mas
    '''
    if 'sigma' in kwargs.keys():
        sigma = kwargs.get('sigma',False)
    else:
        sigma=3

    files   = [eh.image.load_fits(m,aipscc=True) for m in maps]
    fovx        = np.array([m.fovx() for m in files])
    fovy        = np.array([m.fovy() for m in files])
    header  = [read_header(m) for m in maps]
    freq1 = np.round(header[0]['CRVAL3']*1e-9,1)
    freq2 = np.round(header[1]['CRVAL3']*1e-9,1)

    maps_beam=[[h['BMAJ']*np.pi/180,h['BMIN']*np.pi/180,h['BPA']*np.pi/180] for h in header]
    maps_ps = np.array([dm.psize for dm in files])
    naxis1  = (fovx/maps_ps).astype(int)
    naxis2  = (fovy/maps_ps).astype(int)
    ppb=[PXPERBEAM(bp[0],bp[1],pxi) for bp,pxi in zip(maps_beam,maps_ps)]
    noise1=header[0]['NOISE']
    noise2=header[1]['NOISE']

    r2m = 180/np.pi*3.6e6
    if beam=='mean':
        _maj = np.mean([maps_beam[0][0],maps_beam[1][0]])
        _min = np.mean([maps_beam[0][1],maps_beam[1][1]])
        _pos = np.mean([maps_beam[0][2],maps_beam[1][2]])
        sys.stdout.write(' Will use mean beam.\n')
    if beam=='max':
        if np.logical_and(maps_beam[0][0] > maps_beam[1][0],maps_beam[0][1]>maps_beam[1][1]):
            _maj = maps_beam[0][0]
            _min = maps_beam[0][1]
            _pos = maps_beam[0][2]
        elif np.logical_and(maps_beam[0][0] < maps_beam[1][0],maps_beam[0][1]<maps_beam[1][1]):
            _maj = maps_beam[1][0]
            _min = maps_beam[1][1]
            _pos = maps_beam[1][2]
        else:
            print('could not derive max beam.')
            return
        sys.stdout.write(' Will use max beam.\n')
    if beam=='median':
        _maj = np.median([maps_beam[0][0],maps_beam[1][0]])
        _min = np.median([maps_beam[0][1],maps_beam[1][1]])
        _pos = np.median([maps_beam[0][2],maps_beam[1][2]])
        sys.stdout.write(' Will use median beam.\n')
    if beam == 'circ':
        _maj = np.median([maps_beam[0][0],maps_beam[1][0]])
        _min = _maj
        _pos = 0
    if type(beam) == list:
        _maj,_min,_pos = beam
        _maj = r2m
        _min = r2m
        _pos = _pos*180/np.pi

    common_beam = [_maj,_min,_pos]
    sys.stdout.write('Will use beam of ({} , {} ) mas at PA {} degree.\n'.format(_maj*r2m,_min*r2m,_pos*180/np.pi))

    #derive common parameter

    common_ps   = maps_ps.min()
    common_fov  = min([min(x,y) for x,y in zip(fovx,fovy)])
    common_naxis= int(common_fov/common_ps)
    common_ppb  = PXPERBEAM(common_beam[0],common_beam[1],common_ps)

    noise1_r = noise1/ppb[0]*common_ppb
    noise2_r = noise2/ppb[1]*common_ppb


    # regrid and blur the clean maps
    #check for same fov and pixel size
    if np.logical_and(maps_ps[0] != maps_ps[1], fovx[0]!=fovy[1]):

        file1regrid     =   files[0].regrid_image(common_fov, common_naxis, interp='linear')
        file1regridblur =   file1regrid.blur_gauss(common_beam,frac=1)
        file2regrid     = files[1].regrid_image(common_fov, common_naxis, interp='linear')
        file2regridblur = file2regrid.blur_gauss(common_beam, frac=1)

    file1rb = file1regridblur.imarr(pol='I').copy()
    file2rb = file2regridblur.imarr(pol='I').copy()

    mask1 = np.zeros_like(file1rb, dtype=np.bool)
    mask2 = np.zeros_like(file2rb, dtype=np.bool)
    # cut out inner, optically thick part of the image
    if 'npix_x' in kwargs.keys():
        npix_x = kwargs.get('npix_x',False)
        npix_y = kwargs.get('npix_y',False)
        px_min_x = int(common_naxis/2-npix_x)
        px_max_x = int(common_naxis/2+npix_x)
        px_min_y = int(common_naxis/2-npix_y)
        px_max_y = int(common_naxis/2+npix_y)

        px_range_x = np.arange(px_min_x,px_max_x+1,1)
        px_range_y = np.arange(px_min_y,px_max_y+1,1)

        index=np.meshgrid(px_range_y,px_range_x)
        mask1[tuple(index)] = True
        mask2[tuple(index)] = True

    if 'cut_left' in kwargs.keys():
        cut_left = kwargs.get('cut_left',False)
        px_max = int(common_naxis/2.+cut_left)
        px_range_x = np.arange(0,px_max,1)
        mask1
        mask1[:,px_range_x] = True
        mask2[:,px_range_x] = True

    if 'cut_right' in kwargs.keys():
        cut_right = kwargs.get('cut_right',False)
        px_max = int(common_naxis/2-cut_right)
        px_range_x = np.arange(px_max,naxis,1)
        mask1[:,px_range_x] = True
        mask2[:,px_range_x] = True

    if 'radius' in kwargs.keys():
        radius = kwargs.get('radius',False)
        rr,cc = circle(int(len(file1rb)/2),int(len(file1rb)/2),radius)
        mask1[rr,cc] = True
        mask2[rr,cc] = True

    if 'e_maj' in kwargs.keys():
        e_maj = kwargs.get('e_maj',False)
        e_min = kwargs.get('e_min',False)
        e_pa    = kwargs.get('e_pa',False)
        e_xoffset = kwargs.get('e_xoffset',False)
        if e_xoffset!=False:
            x,y = int(len(file1rb)/2)+e_xoffset,int(len(file1rb)/2)
        else:
            x,y = int(len(file1rb)/2),int(len(file1rb)/2)
        if e_pa!=False:
            e_pa = 90-e_pa
        else:
            e_pa = maps_beam[0][2]
        rr,cc =ellipse(y,x,e_maj,e_min,rotation=e_pa*np.pi/180)
        mask1[rr,cc] = True
        mask2[rr,cc] = True

    if 'flux_cut' in kwargs.keys():
        flux_cut = kwargs.get('flux_cut',False)
        mask1[file1>flux_cut*ma.amax(file1rb)] = True
        mask2[file2>flux_cut*ma.amax(file2rb)] = True

    file1rbm = file1rb.copy()
    file2rbm = file2rb.copy()
    file1rbm[mask1]=0
    file2rbm[mask2]=0


    #align image
    # ps covnerted from radperpx to masperpx
    if masked_shift:
        sys.stdout.write('Will derive the shift using the mask during cross-correlation\n')
        file2rb_shift = align(file1rb,file2rb,common_ps*180/np.pi*3.6e6,common_ps*180/np.pi*3.6e6,mask1=~mask1,mask2=~mask2)
    else:
        sys.stdout.write('Will derive the shift using already masked images\n')
        file2rb_shift=align(file1rbm,file2rbm,common_ps*180/np.pi*3.6e6,common_ps*180/np.pi*3.6e6)

################################
    file1_plt = files[0].regrid_image(fovx[0],naxis1[0]).blur_gauss(maps_beam[0],frac=1).imarr(pol='I')
    file2_plt = files[1].regrid_image(fovx[1],naxis1[1]).blur_gauss(maps_beam[1],frac=1).imarr(pol='I')
    file1_plt       *= ppb[0]
    file2_plt       *= ppb[1]
    file1rb_plt = file1rb*common_ppb
    file2rb_plt = file2rb*common_ppb
    file1rbm_plt    = file1rbm*common_ppb
    file2rbm_plt    = file2rbm*common_ppb
    file2rb_shift_plt = apply_shift(file2rb,file2rb_shift['shift'])* common_ppb

    ra=(common_fov/eh.RADPERUAS/1e3/2)-1
    #dec=common_fov/eh.RADPERUAS/1e3/3
    dec = ra*7/10
    ra_min=-ra
    ra_max=ra
    dec_min=-dec
    dec_max=dec
    scale1  = maps_ps[0]*180/np.pi*3.6e6
    scale2  = maps_ps[1]*180/np.pi*3.6e6
    common_scale = common_ps*180/np.pi*3.6e6

    x1=np.linspace(-naxis1[0]*0.5*scale1,(naxis1[0]*0.5-1)*scale1,naxis1[0])
    x2=np.linspace(naxis1[1]*0.5*scale2,-(naxis1[1]*0.5-1)*scale2,naxis1[1])
    xc=np.linspace(common_naxis*0.5*common_scale,-(common_naxis*0.5-1)*common_scale,common_naxis)
    y1=np.linspace(-naxis2[0]*0.5*scale1,(naxis2[0]*0.5-1)*scale1,naxis2[0])
    y2=np.linspace(naxis2[1]*0.5*scale2,-(naxis2[1]*0.5-1)*scale2,naxis2[1])
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

        l1 = '{} GHz original'.format(freq1)
        l2 = '{} GHz original'.format(freq2)
        l3 = '{} GHz regrid $+$ blur'.format(freq1)
        l4 = '{} GHz regrid $+$ blur'.format(freq2)
        label=[l1,l2,l3,l4]

        imax = max([ma.amax(ii) for ii in [file1_plt,file2_plt,file1rb_plt,file2rb_plt]])
        norm = mpl.colors.SymLogNorm(linthresh=level0*1e3,linscale=0.5,vmin=level0*1e3,vmax=0.5*imax*1e3,base=10)
        im1 = ax[0,0].imshow(file1_plt*1e3,cmap=colormap,norm=norm,extent=extent1,zorder=1)
        plotBeam(maps_beam[0][0]*r2m,maps_beam[0][1]*r2m,maps_beam[0][2]*180/np.pi,ra,-dec,ax=ax[0,0])
        im2 = ax[0,1].imshow(file2_plt*1e3,cmap=colormap,norm=norm,extent=extent2,zorder=1)
        plotBeam(maps_beam[1][0]*r2m,maps_beam[1][1]*r2m,maps_beam[1][2]*180/np.pi,ra,-dec,ax=ax[0,1])
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
        f.savefig(plotDir+'{:d}GHz_convolved_with_{:d}GHz.pdf'.format(int(freq2),int(freq1)),bbox_inches='tight')

######################################################  

#   plt.cla()
    if plot_shifted:
        #fig_size=('aanda')
        f = plt.figure(constrained_layout=True)
        f = plt.figure()
        gs = f.add_gridspec(2, 2, hspace=0,wspace=0)
        ax = gs.subplots(sharex='col',sharey='row')

        l1 = '{} GHz regrid $+$ blur'.format(freq1)
        l2 = '{} GHz regrid $+$ blur'.format(freq2)
        l3 = '{0} GHz/ {1} GHz not shifted'.format(freq1,freq2)
        l4 = '{0} GHz/ {1} GHz shifted'.format(freq1,freq2)
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
        ax[1,0].legend([h1[0],h2[0]],['{}GHz'.format(freq1),'{}GHz'.format(freq2)],loc=3,prop={'size':asize})
        #
        cntr1=ax[1,1].contour(np.flipud(file1rb_plt),linewidths=0.5,levels=lev1_r,colors='grey',extent=extentc)
        cntr2=ax[1,1].contour(np.flipud(file2rb_shift_plt),linewidths=0.5,levels=lev2_r,colors='darkblue',extent=extentc,alpha=0.6)
        h1,_ = cntr1.legend_elements()
        h2,_ = cntr2.legend_elements()
        ax[1,1].legend([h1[0],h2[0]],['{}GHz'.format(freq1),'{}GHz'.format(freq2)],loc=3,prop={'size':asize})

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
        f.savefig(plotDir+'shifted_maps_{:d}GHz_{:d}GHz.pdf'.format(int(freq1),int(freq2)),bbox_inches='tight')
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

        a = np.log10(spix2/spix1)/np.log10(freq2/freq1)

        spix_vmin,spix_vmax=-3,5
        sys.stdout.write('\nSpectral index max(alpha)={} - min(alpha)={}\nCutoff {}<alpha<{}\n'.format(ma.amax(a),ma.amin(a),spix_vmin,spix_vmax))
        a[a<spix_vmin] = spix_vmin
        a[a>spix_vmax] = spix_vmax
        a[spix2==noise2*sigma] =spix_vmin

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
        ax.legend([h1[0]],['{} GHz'.format(freq1)],loc=2,prop={'size':asize+1})

        ax.axis('scaled')
        ax.set_xlabel('RA [mas]')
        ax.set_ylabel('Relative Dec [mas]')
        ax.set_xlim(ra_max,ra_min)
        ax.set_ylim(dec_min,dec_max)
        ax.minorticks_on()

        figsize=set_size(fig_size)
        set_corrected_size(f,figsize)

        plt.savefig(plotDir+'spectral_index_between_{:d}_{:d}.pdf'.format(int(freq1),int(freq2)),bbox_inches='tight')
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
