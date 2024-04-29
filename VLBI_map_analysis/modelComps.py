#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table,vstack
from astropy.io import fits,ascii
from astropy.time import Time
import scipy as sp
from glob import glob
import sys,os
from itertools import cycle,chain,compress
from VLBIana.modules.jet_calculus import *
from VLBIana.modules.plot_functions import *
import VLBIana.modules.fit_functions as ff
import ehtim as eh
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.ion()
plt.style.use('pubstyle')

def trim_axs(axs, N):
  """little helper to massage the axs list to have correct length..."""
  axs = axs.flat
  for ax in axs[N:]:
    ax.remove()

  return axs[:N]
def keyfunc(s):
  return [int(''.join(g)) if k else ''.join(g) for k, g in groupby(s, str.isdigit)]

def apply_shift(img,shift):
    offset_image = fourier_shift(np.fft.fftn(img), shift)
    imgalign = np.fft.ifftn(offset_image)
    img2 = imgalign.real

    return img2

class modelComp(object):
    '''
    To read and plot modelfit components
    If you need the initial moFiles array to be sorted: try:
    modFiles = sorted(modFiles,keyfunc)
    It should sort as you typically expect, e.g. 0,1,2,3 or a,b,c

    Instruction:
    modelFile= glob('model/*.fits')
    modF = modelComp(modelFile)
    modF.plot_epoch_xy()

    For sorting by id:
    modF.sort_by_id()

    Then you can use modF.model_sorted for plotting by id
    e.g. as is done in modF.plot_comp_xy(xax='date',yax='FLUX')
    '''
    def __init__(self,modFiles,cleanFiles=None,shift=None,colormap=None,z=None,use_ehtim=True): #**kwargs
       # args = {'cleanFiles':None,
       #         'shift':None,
       #         'colormap':None,
       #         'z':None,
       #         'use_ehtim':True}
       # args.update(kwargs)
        sys.stdout.write('initiate modelComp class.\n')
        self.modFiles =modFiles
        self.model  =   dict()
        self.model_sorted = None
        self.cchead = dict()
        self.ccmap  = dict()
        self.ccmap_shifted = dict()
        self.epochs = []
        self.ids    = []
        self.shift  = None
        self.keys    = [chr(value) for value in range(ord('a'),ord('a')+len(self.modFiles))]
        self.z      = z

        if self.z==None:
            sys.stdout.write('Please provide a redshift if Tb should be calculated, giving the parameter of z when calling the modelComp class.\n')
        else:
            sys.stdout.write('Redshift provided, hence Tb will be calculated.\n')

        if colormap:
            self.colormap = colormap
        else:
            sys.stdout.write('setting colormap inferno')
            self.colormap = 'inferno'
        self.id_colors = []
        self.symbols = []

        for ii,modFile in enumerate(modFiles):
            with fits.open(modFile) as f:
                h = f[0].header
                fnames = h['OBJECT']+'.'+str(np.around(h['CRVAL3']/1e9,decimals=2))+'.'+h['DATE-OBS']
                self.epochs.append(fnames)
                key = self.keys[ii]
                self.model[key] = dict()
                self.model[key]['data_table'] = f[1].data

            self.model[key]['head'] = h
            self.model[key]['data'] = Table(self.model[key]['data_table'])
            self.model[key]['date'] = Time(h['DATE-OBS'])
            self.model[key]['date_obs'] =  self.model[key]['date'].value.split()[0]
            self.model[key]['date'].format = 'mjd'
            self.model[key]['date'] = self.model[key]['date'].value

            self.model[key]['freq'] = np.around(h['CRVAL3']/1e9,2)
            self.model[key]['source'] = h['OBJECT']
            self.model[key]['beam'] = [h['BMAJ'],h['BMIN'],h['BPA']]
            self.model[key]['noise'] = h['NOISE']
            self.model[key]['px_inc'] = h['CDELT2']
            self.model[key]['data']['DELTAX'] *= 3.6e6
            self.model[key]['data']['DELTAY'] *= 3.6e6
            self.model[key]['data']['MAJOR AX'] *= 3.6e6
            self.model[key]['data']['MINOR AX'] *= 3.6e6
            self.model[key]['data']['ratio'] = self.model[key]['data']['MINOR AX'] / self.model[key]['data']['MAJOR AX']
            self.model[key]['data']['DIST'] = np.sign(self.model[key]['data']['DELTAX'])*np.sqrt(self.model[key]['data']['DELTAX']**2 + self.model[key]['data']['DELTAY']**2 )
            self.model[key]['data'].add_column(np.arange(1,len(self.model[key]['data'])+1,1).astype(str)
, name='id', index=0)
            nid = len(self.model[key]['data'])
            self.model[key]['data']['ep'] = nid*[self.keys[ii]]
            self.model[key]['data']['freq'] = nid*[self.model[key]['freq']]
            self.model[key]['data']['cid'] = mpl.colors.to_hex('red')
#include tb here
            if self.z:
                self.model[key]['data']['tb'] = derive_tb(self.model[key]['data']['FLUX'],self.model[key]['freq'],self.z,self.model[key]['data']['MAJOR AX'],self.model[key]['data']['ratio'])
                for ii,value in enumerate(self.model[key]['data']['tb']):
                    if value>1e13:
                        self.model[key]['data']['tb'] [ii] = np.nan
                self.model[key]['data']['logtb'] = np.log10(self.model[key]['data']['tb'])

    #generate a list of the all components
        [self.ids.extend(self.model[key]['data']['id']) for key in self.keys]
        self.ids = np.unique(self.ids)
        self.ids = np.array(sorted(self.ids, key=keyfunc))
        # generate a standard color and symbol for each id
        NUM_COLORS = len(self.ids)
        cmap = mpl.colormaps.get_cmap(self.colormap)
        colors = cmap(np.linspace(0,0.85,NUM_COLORS))

        for i in range(len(colors)):
            self.id_colors.append(mpl.colors.rgb2hex(colors[i]))
        self.id_colors = np.array(self.id_colors)
        self.symbols = np.array(['o','v','*','P','s','D'] * int(np.ceil(NUM_COLORS/6)))

        for key in self.keys:
            for ii,model in enumerate(self.model[key]['data']):
                mask = self.ids==model['id']
                model['cid'] = self.id_colors[mask][0]

        if cleanFiles:
            sys.stdout.write('Load clean fits files to class.\n')
            for i,cleanFile in enumerate(cleanFiles):
                with fits.open(cleanFile) as f:
                    h = f[0].header
                    if not use_ehtim:
                        img = f[0].data
                        if img.ndim == 4:
                            img = img.reshape(img.shape[2],img.shape[3])

                if use_ehtim:
                    file = eh.image.load_fits(cleanFile,aipscc=True)
                    fovx = file.fovx()
                    #fovy = file.fovy()
                    maps_ps = file.psize
                    maps_beam=[h['BMAJ']*np.pi/180,h['BMIN']*np.pi/180,h['BPA']*np.pi/180]
                    naxis1  = int(fovx/maps_ps)
                    ppb = PXPERBEAM(maps_beam[0],maps_beam[1],maps_ps)
                    img = file.regrid_image(fovx,naxis1).blur_gauss(maps_beam,frac=1).imarr(pol='I')
                    img *= ppb
                    img = np.flipud(img)

                key = self.keys[i]
                self.cchead[key] = dict()
                self.cchead[key]['head']    = h
                self.cchead[key]['beam']    = [h['BMAJ']*3.6e6,h['BMIN']*3.6e6,h['BPA']]
                self.cchead[key]['noise']   = h['NOISE']
                self.cchead[key]['px_inc']  = h['CDELT2']
                self.cchead[key]['naxis']   = h['NAXIS1']
                self.cchead[key]['freq']    = np.around(h['CRVAL3']/1e9,2)
                self.cchead[key]['fov']     = self.cchead[key]['px_inc']*self.cchead[key]['naxis']*3.6e6
                self.ccmap[key] = img
 
        if shift:
            sys.stdout.write('shifting model data according to values in shiftFIle.\n')
            shiftFile= shift
            self.shift = Table.read(shiftFile, format='ascii')
            for i,shift in enumerate(self.shift):
                key =self.keys[i]
                self.model[key]['data_shifted'] = self.model[key]['data'].copy()
                self.model[key]['data_shifted']['DELTAX'] += shift['RA']
                self.model[key]['data_shifted']['DELTAY'] += shift['DEC']
                self.model[key]['data_shifted']['DIST'] += np.sqrt(shift['RA']**2 + shift['DEC']**2)
                inc=self.cchead[key]['px_inc']*3.6e6
                self.ccmap_shifted[key] = apply_shift(self.ccmap[key],[shift['DEC']/inc,-shift['RA']/inc])
                self.cchead[key]['shiftDEC']=shift['DEC']

    def update_ids(self):
        ### This does not work if there is a mix of original numbering and changed numbers, giving a mixture of numered strings '1','2' etc and strings starting with a letter 'A1','B4' etc.
        sys.stdout.write('Updatend component id list.\n')
        self.ids = []
        [self.ids.extend(self.model[key]['data']['id']) for key in self.keys]
        self.ids = np.unique(self.ids)
        self.ids = np.array(sorted(self.ids, key=keyfunc))

    def update_cm(self,new_colormap=False):
        if new_colormap==False:
            new_colormap=self.colormap
        else:
            self.colormap = new_colormap
        sys.stdout.write('Updating colormap to {}.\n'.format(self.colormap))
        self.update_ids()
        NUM_COLORS = len(self.ids)
        cmap = mpl.colormaps.get_cmap(self.colormap)
        colors = cmap(np.linspace(0,0.85,NUM_COLORS))
        self.id_colors=[]
        for i in range(len(colors)):
            self.id_colors.append(mpl.colors.rgb2hex(colors[i]))
        self.id_colors = np.array(self.id_colors)
        self.symbols = np.array(['o','v','*','P','s','D'] * int(np.ceil(NUM_COLORS/6)))
        for key in self.keys:
            for ii,model in enumerate(self.model[key]['data']):
                mask = self.ids==model['id']
                model['cid'] = self.id_colors[mask][0]

    def sort_by_id(self):
        sys.stdout.write('Creating class.model_sorted with components sorted by id.\n')
        self.update_ids()

        self.model_sorted = dict()
        if self.shift:
            self.model_sorted_shifted=dict()
        for ID in self.ids:
            date = []
            for i,key in enumerate(self.model):
                if i ==0:
                    self.model_sorted[ID] = Table(dtype=self.model[key]['data'])
                    if self.shift:
                        self.model_sorted_shifted[ID] = Table(dtype=self.model[key]['data_shifted'])
                row = self.model[key]['data'][self.model[key]['data']['id']==ID]
                row_shift = self.model[key]['data_shifted'][self.model[key]['data_shifted']['id']==ID]
                if len(row) > 0:
                    self.model_sorted[ID].add_row(row[0])
                    date.append(self.model[key]['date'])
                    if self.shift:
                        self.model_sorted_shifted[ID].add_row(row_shift[0])

            self.model_sorted[ID]['date']= date
            if self.shift:
                self.model_sorted_shifted[ID]['date']= date
        self.update_cm()


    def change_id(self,old_ids,new_ids):
        '''
        Usage:
            old_ids = ['a_2','b_5','c_6'] is a list of the component ids to be changed. Consisting of the epoch id (ep) and the component id in that epoch (id)
            new_ids = ['a2','a2','b'] a list of names you want to assign to the components. The id of the component in this epoch will then be changed to this value
        '''
        sys.stdout.write('Change model ids according to imput.\n')
        for ii,oids in enumerate(old_ids):
            epoch,cid = oids.split('_')
            #key=self.keys[np.where(np.array(self.eps)==epoch)[0][0]]
            key = epoch
            jj=np.where(self.model[key]['data']['id']==cid)[0][0]
            self.model[key]['data']['id'][jj]=new_ids[ii]
            if self.shift:
                self.model[key]['data_shifted']['id'][jj]= new_ids[ii]
        self.update_cm()
        self.sort_by_id()
###
    def add_comp_data(self,add_comp):
        cID = add_comp['id']
        if 'MINOR AX' not in add_comp.keys():
            add_comp['MINOR AX'] = add_comp['MAJOR AX']
            add_comp['ratio'] = 1
        if cID in self.model_sorted.keys():
            self.model_sorted[cID].add_row(self.model_sorted[cID][0])
            for key in self.model_sorted[cID][-1].keys():
                if key in add_comp.keys():
                    self.model_sorted[cID][-1][key]=add_comp[key]
                else:
                    if isinstance(self.model_sorted[cID][-1][key],np.floating):
                        self.model_sorted[cID][-1][key]= 0.0
            if self.model_sorted[cID][-1]['tb']==0.0:
                self.model_sorted[cID][-1]['tb'] = derive_tb(self.model_sorted[cID][-1]['FLUX'],self.model_sorted[cID][-1]['freq'],self.z,self.model_sorted[cID][-1]['MAJOR AX'],self.model_sorted[cID][-1]['ratio'])
            if self.shift:
                self.model_sorted_shifted[cID].add_row(self.model_sorted_shifted[cID][0])
                for key in self.model_sorted_shifted[cID][-1].keys():
                    if key in add_comp.keys():
                        self.model_sorted_shifted[cID][-1][key]=add_comp[key]
                    else:
                        if isinstance(self.model_sorted_shifted[cID][-1][key],np.floating):
                            self.model_sorted_shifted[cID][-1][key]= 0.0
                if self.model_sorted_shifted[cID][-1]['tb']==0.0:
                    self.model_sorted_shifted[cID][-1]['tb'] = derive_tb(self.model_sorted_shifted[cID][-1]['FLUX'],self.model_sorted_shifted[cID][-1]['freq'],self.z,self.model_sorted_shifted[cID][-1]['MAJOR AX'],self.model_sorted_shifted[cID][-1]['ratio'])

        else:
            sys.stdout.write('Please provide an existing component additional data\n')

###
    def plot_comp_xy(self,xax='DIST',yax='FLUX',out=False,comps=False,line=False,ccolor=False):
        sys.stdout.write('Plot component {} vs {}\n'.format(xax,yax))
        if not self.model_sorted:
            self.sort_by_id()
        if self.shift:
            model_sorted = self.model_sorted_shifted.copy()
        else:
            model_sorted = self.model_sorted.copy()
        if comps:
            model_sorted = {key: model_sorted[key] for key in comps}

        if ccolor:
            ccolors= np.array(len(self.ids)*[ccolor])
        else:
            ccolors = self.id_colors

        fig,ax = plt.subplots(figsize=(10,6))
        if xax == 'freq':
            xlabel='Freq [GHz]'
        else:
            xlabel=xax
        if yax == 'FLUX':
            ylabel='Flux [Jy]'
        elif yax == 'tb':
            ylabel =r'$\log(T_\mathrm{B})$ [K]'
            #ax.set_yscale('log')
            ax.loglog()
        elif yax =='DIST':
            ylabel = 'Distance [mas]'
        elif yax == 'DELTAX':
            ylabel = 'RA [mas]'
        elif yax == 'DELTAY':
            ylabel = 'DEC [mas]'
        else:
            ylabel = yax

        for i,comp in enumerate(model_sorted):
            color = ccolors[self.ids==comp][0]
            symbol = self.symbols[i]
            model_sorted[comp].sort(xax)
            xx = model_sorted[comp][xax]
            yy = model_sorted[comp][yax]
            if line:
                ax.plot(xx,yy,marker=symbol, color = color, label = comp,lw=1)
            else:
                ax.scatter(xx,yy,marker=symbol, color = color, label = comp)

        #ax.set(xlabel=xax, ylabel=yax)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        keys_list = sorted(by_label, key=keyfunc)
        sorted_by_label = {key: by_label[key] for key in keys_list}
        #pos = ax.get_position()
        #ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        #ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        fig.legend(sorted_by_label.values(), sorted_by_label.keys(),loc='upper left', bbox_to_anchor=(0.9,0.98),ncol=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.minorticks_on()
        if not yax=='tb':
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

       # fig.legend(loc=7, fontsize='small')
        fig.tight_layout()
        fig.subplots_adjust(right=0.9, top=0.98)

        if out:
            if type(out)==bool:
                outf = self.model[self.keys[0]]['source']+'_{}_vs_{}.pdf'.format(xax,yax)
            fig.savefig(outf,bbox_inches='tight')
            sys.stdout.write('Plot has been written to {}\n'.format(outf))
        else:
            plt.show()

###

    def plot_epoch_xy(self,xax='DIST',yax='FLUX',plot_all=True,out=True,ylog=False,xlog=False):
        if plot_all:
            nn = len(self.model)
            nx = 4
            ny = int(np.ceil(nn/nx))
        else:
            nx = 1
            ny = 1

        fig,axs = plt.subplots(ny,nx, figsize=(12,8))
        axs   = trim_axs(axs,len(self.model))

        if plot_all:
            for ax,modF in zip(axs,self.model):
                xx = self.model[modF]['data'][xax]
                yy = self.model[modF]['data'][yax]
                if ylog:
                    ax.set_yscale('log')
                if xlog:
                    ax.set_xscale('log')

                ax.scatter(xx,yy)
                ax.minorticks_on()
                ax.set(xlabel=xax,ylabel=yax)
                ax.annotate('{} - {:.1f} GHz - {}'.format(self.model[modF]['source'],self.model[modF]['freq'],self.model[modF]['date']),xy=(0.03,0.9),xycoords='axes fraction',size=8)

            plt.tight_layout(pad=0.2,w_pad=0.2,h_pad=0.2)
            fig.subplots_adjust(right=0.95, top=0.98)

            if out:
                if type(out)==bool:
                    outf = self.model[modF]['source']+'_components.pdf'
                fig.savefig(outf,bbox_inches='tight')
                sys.stdout.write('Plot has been written to {}\n'.format(outf))
            else:
              plt.show()
            plt.cla()

    def plot_evolution_map(self,sigma=3,fig_size='screen',ra=False,dec=False,saveFile=False,plot_color=False,cntr_color=False,cntr_lw=False,ccolor=False,out=True,plot_id=False,plot_below=False,shifted=False,plot_cmp_arrow=False,plot_label=True,plot_ontop=False,plot_cmp_name=False,epochs=False,plot_cmp_lines=False,plot_comps=False,plot_title=False,exclude_model=False,cmp_name_color=None,nx=1):
        """ Plot evolution of VLBI maps
        """
        sys.stdout.write('Plot modelcomponents over clean maps\n')
        if epochs:
            ids         = []
            [ids.extend(self.model[key]['data']['id']) for key in epochs]
            ids = np.unique(ids)
            ids = np.array(sorted(ids, key=keyfunc))
            id_colors = np.array([self.id_colors[ii] for ii,iids in enumerate(self.ids) if iids in ids])
            sys.stdout.write('Plotting maps and components for epochs {}\n'.format(epochs))
        else:
            epochs = self.keys
            ids = self.ids
            id_colors = self.id_colors
        nn = len(epochs)
        nx = nx
        ny = int(np.ceil(nn/nx))

        if not cntr_color:
                cntr_color = 'grey'
        if not cntr_lw:
            cntr_lw=1

        if ccolor:
            sys.stdout.write('Set user component color\n')
            ccolors= len(ids)*[ccolor]
        else:
            ccolors = id_colors
        if dec:
            if type(dec)==list:
                if len(dec)>2:
                    if len(dec) > len(epochs):
                        sys.stdout.write('please give only 1 parameter for each fits-image ra and dec\n')
                        return
            else:
                ra  = np.mean([self.cchead[key]['fov']/3. for key in epochs])
                dec = [self.cchead[key]['fov']/5. for key in epochs]

        Dra = [-ra,ra]
        shifty = [0]
        nk = len(epochs)
        for kk,keys in enumerate(epochs):
            if kk>0:
                shifty.append(shifty[kk-1]-dec[kk-1]*1.2)
           # if kk>1:
            #    shifty.append(shifty[kk-1]-2*dec[kk-1])

        if plot_ontop:
            Ddec = [-dec[0],dec[0]]
        else:
            Ddec= [shifty[-1]-dec[-1]*2,dec[0]]
            #Ddec = [-np.sum(dec),dec[0]]
        if plot_color:
            plt.style.use('dark_background')
        xsize=6
        if plot_ontop:
            ysize=dec[0]*xsize/ra
        else:
            ysize = np.sum(dec)*xsize/ra
        fig,ax = plt.subplots(1,1,figsize=(xsize,ysize), constrained_layout=True)
        ax.set_ylabel('Frequency [GHz]')#,fontsize=12)
        ax.set_xlabel('RA [mas]')#,fontsize=12)

        lines   = []
        labels  = []
        xmin,xmax = [],[]
        ax.axis('scaled')
        ax.set_xlim(Dra)
        ax.set_ylim(Ddec)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if plot_title:
            #ax.set_title(plot_title,size=12)
            bbox_props= dict(boxstyle='round',alpha=0.2)
            ax.annotate(plot_title, xy=(0.6,0.95),xycoords='axes fraction',fontsize=16,bbox=bbox_props,zorder=20)

        else:
            ax.set_title('{} - {}'.format(self.model[epochs[0]]['source'],self.model[epochs[0]]['date_obs']),size=10)

        ax.invert_xaxis()

        ####################
        # setting all parameters for plotting a clean image
        #####################
        mapshift=0
        for kk,key in enumerate(epochs):
            modelh= self.model[key]
            clean = self.cchead[key]
            if shifted:
                model = modelh['data_shifted']
            else:
                model = modelh['data']
            if shifted:
                ccmap = self.ccmap_shifted[key].copy()
            else:
                ccmap =  self.ccmap[key].copy()
            inc=clean['px_inc']*3.6e6

            if not plot_ontop:
                if key!=epochs[0]:
                    model['DELTAY'] += shifty[kk]
                    model['DIST'] += shifty[kk]

            if plot_ontop:
                sdec =dec[kk]
            else:
                sdec = shifty[kk]
            # create loglevels
            scale   = -clean['px_inc']*3.6e6
            sigma   = sigma
            level0  = clean['noise']*sigma
            lev=[]
            for i in range(0,10):
                lev.append(level0*2**i)
            xx  = np.linspace(-clean['naxis']*0.5*scale,(clean['naxis']*0.5-1)*scale,clean['naxis'])
            yy  = np.linspace(clean['naxis']*0.5*scale,-(clean['naxis']*0.5-1)*scale,clean['naxis'])
            if plot_color:
                ymask = np.logical_and(-dec[kk]<=yy,yy<=dec[kk])
            else:
                if kk==0:
                    ymask = np.logical_and(-dec[kk]<=yy,yy<=dec[kk])
                else:
                    ymask = np.logical_and(-dec[kk]*1.5<=yy,yy<=dec[kk]*1.5)
            YY = yy[ymask]
            yy=YY
            ccmap=ccmap[ymask,:]


            vmax=ma.amax(ccmap)
            norm = mpl.colors.SymLogNorm(linthresh=level0,linscale=0.05,vmin=level0,vmax=vmax,base=np.e)

            ##################
            # Plotting
            ###################
            plotBeam(clean['beam'][0],clean['beam'][1],clean['beam'][2],ra,sdec-dec[kk]*0.4,ax)

            if plot_ontop:
                colorsc =['red','blue','green','black']
                cntr_color = colorsc[kk]
                extent = np.max(xx),np.min(xx),np.min(yy),np.max(yy)
            else:
                if plot_color:
                    if(kk==0):
                        extent = max(xx),min(xx),min(yy)+shifty[kk],max(yy)+shifty[kk]
                    else:
                        extent = max(xx),min(xx),2*min(yy)+shifty[kk],shifty[kk]
                else:
                    extent = max(xx),min(xx),min(yy)+shifty[kk],max(yy)+shifty[kk]


            cntr = ax.contour(ccmap,linewidths=cntr_lw,levels=lev,colors=cntr_color,alpha=1,extent=extent)
            if plot_color:
                im = ax.imshow(ccmap,cmap=self.colormap,extent=extent,origin='lower', interpolation='gaussian')
                im.set_norm(norm)

            if plot_comps:
                if key==exclude_model:
                    sys.stdout.write('will not plot model for map {}'.format(key))
                else:
                    model.sort('id')
                    Mx  = model['DELTAX']
                    My  = model['DELTAY']
                    Mposa   = model['POSANGLE']
                    Mmaj    = model['MAJOR AX']
                    Mmin    = model['MINOR AX']
                    for j,MMx in enumerate(Mx):
                        epid=model['id'][j]
                        ccolor = ccolors[ids==epid][0]
                        e_comp = Ellipse([Mx[j],My[j]],Mmaj[j],Mmin[j],-Mposa[j]+90, color=ccolor, zorder=2, fill=False,lw=0.5)
                        ax.add_artist(e_comp)
                        maj1_x = Mx[j]-np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                        maj1_y = My[j]+np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                        maj2_x = Mx[j]+np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                        maj2_y = My[j]-np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5

                        min1_x = Mx[j]-np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                        min1_y = My[j]+np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                        min2_x = Mx[j]+np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                        min2_y = My[j]-np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                        if plot_cmp_lines==key:
                            xmin.append(min1_x.copy())
                            xmax.append(min2_x.copy())
                        if maj1_y==maj2_y:
                            ax.plot(maj1_x,maj1_y, color = ccolor,marker='+', markersize=8,lw=1,label=model['id'][j])
                        elif Mmaj[j] < clean['beam'][0]/10.:
                            ax.plot((maj1_x+maj2_x)/2.,(maj1_y+maj2_y)/2., color = ccolor,marker='+', markersize=10,lw=1,label=model['id'][j])
                        else:
                            ax.plot([maj1_x,maj2_x],[maj1_y,maj2_y], color = ccolor, lw = 0.8,label=model['id'][j])
                            ax.plot([min1_x,min2_x],[min1_y,min2_y], color = ccolor, lw = 0.8)
                        if plot_cmp_arrow:
                            if cmp_name_color:
                                cmp_color = cmp_name_color
                            else:
                                cmp_color = ccolor

                            if j % 2 ==0:
                                yy=25
                            else:
                                yy=40
                            bbox_props= dict(boxstyle='round',fc='w')
                            ax.annotate('{}'.format(model['id'][j]),xy=(Mx[j],My[j]), xycoords='data', xytext=(-5, yy), textcoords='offset points',arrowprops=dict(arrowstyle="->", connectionstyle="arc3",relpos=(0.7,0.5),shrinkA=2,shrinkB=0.1), color=cmp_color,fontsize=9,bbox=bbox_props,zorder=20)
                        elif plot_cmp_name:
                            if cmp_name_color:
                                cmp_color = cmp_name_color
                            else:
                                cmp_color = ccolor
                            if j % 2 ==0:
                                yy=15
                            else:
                                yy=30
                            bbox_props= dict(boxstyle='round',alpha=0.2)
                            ax.annotate('{}'.format(model['id'][j]),xy=(Mx[j],My[j]), xycoords='data', xytext=(-5, yy), textcoords='offset points',color=cmp_color,fontsize=7,bbox=bbox_props,zorder=20)



# The following lines sort the labels and handles alphabetically 
            if plot_label:
                handles, Labels = ax.get_legend_handles_labels()
                labels.extend(Labels)
                lines.extend(handles)
        for kk,key in enumerate(epochs):
            if plot_cmp_lines==key:
                modelh= self.model[key]
                if shifted:
                    model = modelh['data_shifted']
                else:
                    model = modelh['data']
                for j,(x1,x2) in enumerate(zip(xmin,xmax)):
                    epid=model['id'][j]
                    ccolor = ccolors[ids==epid][0]
                    ax.axvline(x = (x1+x2)/2., color = ccolor,lw=0.8,ls='--',alpha=0.5)

        if plot_label:
            by_label = dict(zip(labels, lines))
            keys_list = sorted(by_label, key=keyfunc)
            sorted_by_label = {key: by_label[key] for key in keys_list}
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
            fig.legend(sorted_by_label.values(), sorted_by_label.keys(),loc='upper right', bbox_to_anchor=(1.15,0.88),ncol=1,prop = { "size": 9})
# sorting done, legend plottet
        tlabel = ['{}'.format(self.model[key]['freq']) for key in epochs]
        plt.yticks(shifty,tlabel,fontsize=9)
        plt.xticks(fontsize=10)
        plt.tight_layout()#pad=0.2,w_pad=0.2,h_pad=0.2)
        fig.subplots_adjust(right=0.95, top=0.98)

        if out:
            fig_size='aanda'
            figsize=set_size(fig_size)
            #set_corrected_size(fig,figsize)
            if type(out)==bool:
                outf = modelh['source']+'_map+model_evolution.pdf'
            elif type(out)==str:
                outf = out
            if shifted:
                outf = ''.join(outf.split('.')[:-1])+'_shifted.pdf'
            fig.savefig(outf,bbox_inches='tight')
            sys.stdout.write('Plot has been written to {}\n'.format(outf))
        else:
            plt.show()
        plt.clf()
        return fig,ax



    def overplot_model(self,sigma=3,fig_size='screen',ra=False,dec=False,saveFile=False,plot_model=True,plot_color=False,cntr_color=False,cntr_lw=False,ccolor=False,out=True,plot_id=True,shifted=False,plot_cmp_arrow=False,plot_label=True):
        """I changed alot here to try to plot all maps with the same colorbar. However, that does not look nice for the data I tested. I have to continue work on this.
        """
        sys.stdout.write('Plot modelcomponents over clean maps\n')
#        if plot_all:
        nn = len(self.model)
        if nn>6:
            nx = 4
        elif nn==6:
            nx = 3
        else:
            nx = 1
        ny = int(np.ceil(nn/nx))

        if not cntr_color:
            cntr_color = 'grey'
        if not cntr_lw:
            cntr_lw=1

        if ccolor:
            sys.stdout.write('Set user component color\n')
            ccolors= len(self.ids)*[ccolor]
        else:
            ccolors = self.id_colors
        load_gs = False
        if not dec:
            ra  = np.mean([self.cclean[key]['fov']/3. for key in self.keys()])
            dec = np.mean([self.cclean[key]['fov']/5. for key in self.keys()])
        RA = ra
        DEC = dec
        if type(dec)==list:
            if len(dec)>2 or np.logical_and(len(dec)==2, len(self.keys)==2):
                if len(dec) > len(self.keys):
                    sys.stdout.write('please give only 1 parameter for each fits-image ra and dec\n')
                    return
                load_gs = True
                gs_heights= dec
            elif len(dec)==2 and len(self.keys)==1:
                Dra  = ra
                Ddec = dec
        else:
            Dra = [-ra,ra]
            Ddec= [-dec,dec]

        xsize = 6
        ysize = 3
        xxs = xsize*nx
        yys = np.mean(dec)*xxs/np.mean(ra)
        yys = ysize*ny

        fig,ax = plt.subplots(ny,nx, figsize=(xxs,yys))
        ax   = trim_axs(ax,len(self.model))
        #fig = plt.figure(constrained_layout = True)
        #gs = fig.add_gridspec(ny,nx, hspace=0, wspace=0)
        #ax = gs.subplots(sharex = 'col')

        vmin = ma.amin([self.cchead[key]['noise'] for key in self.keys])*1e3
        vmax = ma.amax([self.ccmap[key] for key in self.keys])*1e3
        level00=vmin*min(sigma)
        #lev = []
        #for i in range(0,10):
        #    lev.append(level0*2**i)
        #norm = mpl.colors.SymLogNorm(linthresh=level0,linscale=0.5,vmin=level0,vmax=0.5*vmax,base=10)


        lines   = []
        labels  = []
        kk=0
        ####################
        # setting all parameters for plotting a clean image
        #####################
        for aa,key in zip(ax.flat,self.keys):
            modelh= self.model[key]
            clean = self.cchead[key]
            if shifted:
                model = self.model[key]['data_shifted']
            else:
                model = self.model[key]['data']
            if shifted:
                ccmap = self.ccmap_shifted[key]
            else:
                ccmap =  self.ccmap[key]
            ccmap *=1e3
            if len(sigma)>1:
                _sigma = sigma[kk]
            else:
                _sigma   = sigma
            #vmin = clean['noise']*1e3
            #vmax = ma.amax(ccmap)

            level0  = vmin*_sigma
            lev=[-level0]
            for i in range(0,10):
                lev.append(level0*2**i)
#            vmax=ma.amax(ccmap)
            #ccmap[ccmap <=level0] = level0
            norm = mpl.colors.SymLogNorm(linthresh=level00,linscale=0.5,vmin=level00,vmax=vmax,base=10)
            #sys.stdout.write('vmax={} ; vmin={}\n lev={} \n'.format(vmax,vmin,lev))

            if load_gs:
                Dra = [-ra[kk],ra[kk]]
                Ddec = [-dec[kk],dec[kk]]
                RA = ra[kk]
                DEC = dec[kk]

            scale   = -clean['px_inc']*3.6e6
            xx  = np.linspace(-clean['naxis']*0.5*scale,(clean['naxis']*0.5-1)*scale,clean['naxis'])
            yy  = np.linspace(clean['naxis']*0.5*scale,-(clean['naxis']*0.5-1)*scale,clean['naxis'])
            extent = np.max(xx),np.min(xx),np.min(yy),np.max(yy)


            ##################
            # Plotting
            ###################
            #f,aa = plt.subplots()
            aa.axis('scaled')
            aa.set_xlim(Dra)
            aa.set_ylim(Ddec)
            aa.set_aspect(1)
            aa.set_adjustable("datalim")
            aa.invert_xaxis()
            print(DEC)
            plotBeam(clean['beam'][0],clean['beam'][1],clean['beam'][2],RA,-DEC,aa,color='white')

            cntr = aa.contour(ccmap,linewidths=cntr_lw,levels=lev[1:],colors=cntr_color,alpha=1,extent=extent)
            if plot_color:
                im = aa.imshow(ccmap,cmap=self.colormap,extent=extent,origin='lower', interpolation='gaussian')
                im.set_norm(norm)
                plt.style.use('dark_background')
            if plot_model:
                Mx  = model['DELTAX']
                My  = model['DELTAY']
                Mposa   = model['POSANGLE']
                Mmaj    = model['MAJOR AX']
                Mmin    = model['MINOR AX']
                for j,xx in enumerate(Mx):
                    epid=model['id'][j]
                    ccolor = ccolors[self.ids==epid][0]
                    e_comp = Ellipse([Mx[j],My[j]],Mmaj[j],Mmin[j],-Mposa[j]+90, color=ccolor, zorder=2, fill=False,lw=0.5)
                    aa.add_artist(e_comp)
                    maj1_x = Mx[j]-np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                    maj1_y = My[j]+np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                    maj2_x = Mx[j]+np.sin(-np.pi/180*Mposa[j])*Mmaj[j]*0.5
                    maj2_y = My[j]-np.cos(-np.pi/180*Mposa[j])*Mmaj[j]*0.5

                    min1_x = Mx[j]-np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                    min1_y = My[j]+np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                    min2_x = Mx[j]+np.sin(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                    min2_y = My[j]-np.cos(-np.pi/180*(Mposa[j]+90))*Mmin[j]*0.5
                    if maj1_y==maj2_y:
                        aa.plot(maj1_x,maj1_y, color = ccolor,marker='+', markersize=5,lw=1,label=model['id'][j])
                    else:
                        aa.plot([maj1_x,maj2_x],[maj1_y,maj2_y], color = ccolor, lw = 1,label=model['id'][j])
                        aa.plot([min1_x,min2_x],[min1_y,min2_y], color = ccolor, lw = 1)
                    if plot_cmp_arrow:
                        aa.annotate('{}'.format(model['id'][j]),xy=(Mx[j],My[j]), xycoords='data', xytext=(-5, (-1)**j*40), textcoords='offset points',arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color=ccolor,fontsize=8)


            aa.set(xlabel='RA [mas]', ylabel='DEC [mas]')
            aa.minorticks_on()
            aa.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            aa.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            aa.set_title('{} - {:.1f} GHz - {}'.format(modelh['source'],modelh['freq'],modelh['date_obs']),size=12)
# The following lines sort the labels and handles alphabetically 
            if plot_label:
                handles, Labels = aa.get_legend_handles_labels()
                labels.extend(Labels)
                lines.extend(handles)
            kk+=1

        if plot_label:
            by_label = dict(zip(labels, lines))
            keys_list = sorted(by_label, key=keyfunc)
            sorted_by_label = {key: by_label[key] for key in keys_list}
            pos = aa.get_position()
            aa.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
            #aa.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            fig.legend(sorted_by_label.values(), sorted_by_label.keys(),loc='upper left', bbox_to_anchor=(1,0.98),ncol=4)
            #secx = aa.secondary_xaxis('left',)
            #secx.set_xlabel(args['Freq'])
# sorting done, legend plottet

        # set axis, labels, etc.
        cbar = fig.colorbar(im, ax=ax, orientation='vertical',fraction=0.15,pad=0.15)
        cbar.set_label(r'$S$ [mJy/beam]')

        plt.tight_layout()#pad=0.2,w_pad=0.2,h_pad=0.2)
        fig.subplots_adjust(right=0.95, top=0.98)

        if out:
            if type(out)==bool:
                outf = modelh['source']+'_Model_overplot.pdf'
            if shifted:
                outf = modelh['source']+'_Model_overplot_shifted.pdf'
            fig.savefig(outf,bbox_inches='tight')
            sys.stdout.write('Plot has been written to {}\n'.format(outf))
        else:
            plt.show()
        plt.clf()
        return fig,ax

    def plot_all(self,xax='date',yax='DIST',out=False):
        fig,ax = plt.subplots(figsize=(12,8))
        dates = []
        ymin = 0
        ymax = 0
        for modF in self.model:
            dates.append(self.model[modF]['date'])
            if np.min(self.model[modF]['data'][yax]) < ymax :
                ymin = np.min(self.model[modF]['data'][yax])
            if np.max(self.model[modF]['data'][yax]) > ymax :
                ymax = np.max(self.model[modF]['data'][yax])
        xmin = np.min(dates) - 0.5
        xmax = np.max(dates) + 0.5
        ax.axis([xmin,xmax,ymax,ymin])
        print([xmin,xmax,ymin,ymax])
        print(dates)
        for i,modF in enumerate(self.model):
            ax.scatter([dates[i]]*len(self.model[modF]['data'][yax]),self.model[modF]['data'][yax])
        ax.minorticks_on()
        if xax=='date':
            ax.set(xlabel='Date [MJD]',ylabel='Distance [mas]')
        else:
            ax.set(xlabel=xax,ylabel='Distance [mas]')
        plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)

        if out:
           if type(out)==bool:
               outf = self.model[modF]['source']+'_all.pdf'
           fig.savefig(outf,bbox_inches='tight')
           sys.stdout.write('Plot has been written to {}\n'.format(outf))
        else:
            plt.show()
        plt.cla()

    def fit_comp_spectrum(self,add_data=False,plot_areas=False,plot_all_components=False,comps=False,exclude_comps=False,ccolor=False,out=True,fluxerr=False,fit_free_ssa=False,plot_fit_summary=False,annotate_fit_results=True):
        '''
        If adding data please provide a dict as add_data={'flux':xxx, 'freq':yyy,'id':zzz})
        '''
        sys.stdout.write('Fit component spectrum\n')
        if not self.model_sorted:
            self.sort_by_id()
        if comps:
            comps_name = comps
        else:
            comps = self.ids
            comps_name = ['All']
        if exclude_comps:
            comps = [key for key in comps if key not in exclude_comps]
            comps_name = ['All-'+''.join([i for i in exclude_comps if '/' not in i])]
        #model_sorted = {key: self.model_sorted[key] for key in comps}
        model_sorted = {key: self.model_sorted_shifted[key] for key in comps}

        del_comps = []
        for comp in model_sorted:
            if len(model_sorted[comp])<2:
                del_comps.append(comp)
        for dd in del_comps:
            del model_sorted[dd]

        if ccolor:
            ccolors= np.array(len(self.ids)*[ccolor])
        else:
            ccolors = self.id_colors
        if add_data:
            model_sorted[add_data['id']].add_row(model_sorted[add_data['id']][0])
            model_sorted[add_data['id']][-1]['FLUX']=add_data['flux']
            model_sorted[add_data['id']][-1]['freq']=add_data['freq']
            if 'major' in add_data.keys():
                model_sorted[add_data['id']][-1]['MAJOR AX']=add_data['major']
                model_sorted[add_data['id']][-1]['MINOR AX']=add_data['minor']

        NCOMP = len(model_sorted)
        if NCOMP==1:
            nx=1
        elif NCOMP<5:
            nx=2
        elif NCOMP<13:
            nx = 3
        else:
            nx = 4
        ny= int(np.ceil(NCOMP/nx))
        print(ny)
        xsize =5
        xxs = nx*xsize
        ysize =3 
        yys= ysize*ny

        fig,axs = plt.subplots(ny,nx, figsize=(xxs,yys))

        xlabel = 'Frequency [GHz]'
        ylabel = 'Flux density [Jy]'
        if NCOMP>1:
            axs = trim_axs(axs,NCOMP)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
        else:
            axs = [axs]
            axs[0].set(xlabel=xlabel, ylabel=ylabel)
# Doing the fit
        sn_p,sn_sd,sn_ch2,an_out,pl_p,pl_sd,pl_ch2,pl_out=[],[],[],[],[],[],[],[]
        fit=[]
        CompSN,CompPL,num,Sm,athin,athick,alpha,chi2SN,chi2PL,numE,SmE,athinE,athickE,alphaE=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
#        ffa_p,ffa_sd,ffa_ch2 = [],[],[]

        for i,comp in enumerate(model_sorted):
            cflux = model_sorted[comp]['FLUX']
            if fluxerr:
                if comp==fluxerr['comp']:
                    cfluxerr = fluxerr['error']*cflux.copy()
                    cfreq = fluxerr['freq']
                else:
                    cfluxerr = 0.15*cflux.copy()
                    cfreq = model_sorted[comp]['freq']
            cid = model_sorted[comp]['id']

            print('Fit Powerlaw to Comp '+cid[0])
            pl_x0 = np.array([np.mean(cflux),-1])
            beta,sd_beta,chi2,pl_out = ff.odr_fit(ff.powerlaw,[cfreq,cflux,cfluxerr],pl_x0,verbose=1)
            pl_p.append(beta)
            pl_sd.append(sd_beta)
            pl_ch2.append(chi2)

#            print('Fit FFA to Comp '+cid[0])
#            ffa_x0 = np.array([10,np.mean(cflux),1])
#            beta,sd_beta,chi2,ffa_out = ff.odr_fit(ff.ffa,[cfreq,cflux,cfluxerr],ffa_x0,verbose=1)
#            ffa_p.append(beta)
#            ffa_sd.append(sd_beta)
#            ffa_ch2.append(chi2)

            #fit Snu
            print('Fit SSA to Comp '+cid[0])
            if fit_free_ssa:
                sn_x0 = np.array([20,np.max(cflux),3,-1])
            else:
                sn_x0 = np.array([20,np.max(cflux),-1])
            if cid[0]=='A15':
                sn_x0[0]=100
            if cid[0]=='C1':
                sn_x0[0] = 10
            if fit_free_ssa:
                beta,sd_beta,chi2,sn_out = ff.odr_fit(ff.Snu,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)
            else:
                beta,sd_beta,chi2,sn_out = ff.odr_fit(ff.Snu_real,[cfreq,cflux,cfluxerr],sn_x0,verbose=1)
            sn_p.append(beta)
            sn_sd.append(sd_beta)
            sn_ch2.append(chi2)

            if np.logical_and(sn_ch2[i]>pl_ch2[i],pl_out.info<5):
                sys.stdout.write('Power law fits better\n')
                CompPL.append(cid[0])
                alpha.append(pl_p[i][1])
                alphaE.append(pl_sd[i][1])
                chi2PL.append(pl_ch2[i])
                fit.append('PL')
            elif np.logical_and(pl_ch2[i]>sn_ch2[i],sn_out.info<5):
                if fit_free_ssa:
                    sys.stdout.write('ssa spectrum fits better\n')
                    CompSN.append(cid[0])
                    num.append(sn_p[i][0])
                    Sm.append(sn_p[i][1])
                    athin.append(sn_p[i][3])
                    athick.append(sn_p[i][2])
                    chi2SN.append(sn_ch2[i])
                    numE.append(sn_sd[i][0])
                    SmE.append(sn_sd[i][1])
                    athinE.append(sn_sd[i][3])
                    athickE.append(sn_sd[i][2])
                    fit.append('SN')
                else:
                    sys.stdout.write('ssa spectrum fits better\n')
                    CompSN.append(cid[0])
                    num.append(sn_p[i][0])
                    Sm.append(sn_p[i][1])
                    athin.append(sn_p[i][2])
                    chi2SN.append(sn_ch2[i])
                    numE.append(sn_sd[i][0])
                    SmE.append(sn_sd[i][1])
                    athinE.append(sn_sd[i][2])
                    fit.append('SN')
                    athick.append(2.5)
                    athickE.append(0.0)

            else:
                sys.stdout.write('NO FIT WORKED, use power law\n')
                CompPL.append(cid[0])
                alpha.append(pl_p[i][1])
                alphaE.append(pl_sd[i][1])
                chi2PL.append(pl_ch2[i])
                fit.append('PL')
        namesSN=['Comp','num','numE','Sm','SmE','athin','athinE','athick','athickE','Chi2']
        SN = Table([CompSN,np.round(num,2),numE,np.round(Sm,2),SmE,np.round(athin,2),athinE,np.round(athick,2),athickE,chi2SN],names=namesSN)
        print(SN)
        namesPL=['Comp','alpha','alphaE','Chi2']
        PL = Table([CompPL,np.round(alpha,2),alphaE,chi2PL],names=namesPL)
        print(PL)
        savefile ='component_spectrum_fit'
        ascii.write(SN,savefile+'_SN.tex',formats=dict.fromkeys(namesSN[1:], '%.2f'),overwrite=True,format='latex')
        ascii.write(PL,savefile+'_PL.tex',formats=dict.fromkeys(namesPL[1:], '%.2f'),overwrite=True,format='latex')
        exponent = -2
        ymin=float('1e{}'.format(exponent))
        props = dict(boxstyle='round',fc='w',alpha=0.5)

        #doing the plotting
        i=0
        for ax,comp in zip(axs,model_sorted):
            cflux = model_sorted[comp]['FLUX']
            if fluxerr:
                if comp==fluxerr['comp']:
                    cfluxerr = fluxerr['error']*cflux.copy()
                    cfreq = fluxerr['freq']
                else:
                    cfluxerr = 0.15*cflux.copy()
                    cfreq = model_sorted[comp]['freq']

            cid = model_sorted[comp]['id']

            pl_x0 = np.array([np.min(cflux),-1])


#            color = ccolors[self.ids==comp][0]
#            symbol = self.symbols[i]

            ax.set_xscale('log')
            ax.set_yscale('log')
            #ax.axis(axe_ratio)
            ax.minorticks_on()
            ax.errorbar(cfreq,cflux,cfluxerr,marker='.',ms=3,ls='none',color='red',elinewidth=0.8,label='Comp {}'.format(cid[0]))
            xr=np.arange(1,300,0.01)
#            ax.plot(xr,ff.ffa(ffa_p[i],xr),'g',lw=0.5)

            if fit[i]=='PL':
                textstr = '\n'.join((
                    r'$\alpha={:.2f}\pm{:.2f}$'.format(pl_p[i][1],pl_sd[i][1]),
                ))
                ax.annotate(textstr, xy=(0.05,0.1),xycoords='axes fraction',fontsize=8,bbox=props)
                ax.plot(xr,ff.powerlaw(pl_p[i],xr),'k',lw=0.5)
                y1 = ff.powerlaw(pl_p[i]-pl_sd[i],xr)
                y2 = ff.powerlaw(pl_p[i]+pl_sd[i],xr)
                ax.fill_between(xr,y1,y2,alpha=0.3)
            elif fit[i]=='SN':
                if fit_free_ssa:
                    textstr = '\n'.join((
                        r'$\nu_m={:.2f}$'.format(sn_p[i][0]),
                        r'$S_m={:.2f}$'.format(sn_p[i][1]),
                        r'$\alpha_{{thin}}={:.2f}$'.format(sn_p[i][3]),
                        r'$\alpha_{{thick}}={:.2f}$'.format(sn_p[i][2]),
                        r'$\chi_\mathrm{{red}}^2={:.2f}$'.format(sn_ch2[i])
                    ))
                else:
                    textstr = '\n'.join((
                        r'$\nu_m={:.2f}$'.format(sn_p[i][0]),
                        r'$S_m={:.2f}$'.format(sn_p[i][1]),
                        r'$\alpha_{{thin}}={:.2f}$'.format(sn_p[i][2]),
                        r'$\chi_\mathrm{{red}}^2={:.2f}$'.format(sn_ch2[i])
                    ))

                if annotate_fit_results:
                    ax.annotate(textstr, xy=(0.05,0.1),xycoords='axes fraction',fontsize=8,bbox=props)
                sn_low = sn_p[i]-sn_sd[i]
                sn_up = sn_p[i]+sn_sd[i]
                for jj,SNL in enumerate(sn_low[:2]):
                    if SNL <0:
                        sys.stdout.write('Uncertainties for SN fit for comp{} large, limit peak flux and freq \n'.format(comp))
                        if jj==0:
                            sn_low[jj] = 0.1
                        if jj == 1:
                            sn_low[jj] = ymin
                if fit_free_ssa:
                    ax.plot(xr,ff.Snu(sn_p[i],xr),'k',lw=0.5)
                    y1 = ff.Snu(sn_low,xr)
                    y2 = ff.Snu(sn_up,xr)
                else:
                    ax.plot(xr,ff.Snu_real(sn_p[i],xr),'k',lw=0.5)
                    y1 = ff.Snu_real(sn_low,xr)
                    y2 = ff.Snu_real(sn_up,xr)

                ax.fill_between(xr,y1,y2,alpha=0.2)

            ax.set_xlim(1,300)
            ax.set_ylim(ymin,5)

            ax.set_yticks([0.001,0.01,0.1,1])
            ax.set_xticks([1,10,100])
            ax.tick_params(axis="x",direction="in")
            ax.tick_params(axis="y",direction="in")
            ax.legend(loc=1)
            i+=1
        fig.subplots_adjust(right=0.8, top=0.98, left=0.06,bottom=0.06)
        #set_corrected_size(fig,figsize)
        if out:
            if type(out)==bool:
                outf = self.model[self.keys[0]]['source']+'_spectrum_fit_components{}.pdf'.format(''.join(comps_name))
            elif type(out)==str:
                outf = out
            fig.savefig(outf,bbox_inches='tight')
            sys.stdout.write('Plot has been written to {}\n'.format(outf))
        else:
            plt.show()

#########3
################
        if plot_fit_summary:
            xax = 'DIST'
            xlabel='Distance [mas]'
            yax = ['num','Sm','athin']

            ylabel = [r'$\nu_\mathrm{m}$ [GHz]',r'$S_\mathrm{m}$ [Jy]',r'$\alpha_\mathrm{thin}$']
            for jj,YY in enumerate(yax):
                xx,xxE = [],[]
                yy,yyE = [],[]
                color,symbol,cid =[],[],[]
                for i,comp in enumerate(model_sorted):
                    if fit[i]=='SN':
                        color.append(ccolors[self.ids==comp][0])
                        symbol.append(self.symbols[i])
                        model_sorted[comp].sort(xax)
                        xx.append(np.mean(model_sorted[comp][xax]))
                        xxE.append(np.std(model_sorted[comp][xax]))
                        yy.append(sn_p[i][jj])
                        yyE.append(sn_sd[i][jj])
                        if yyE[-1]>abs(yy[-1]):
                            yyE[-1]=abs(yy[-1])
                            print(yyE[-1])
                        cid.append(model_sorted[comp]['id'][jj])
                        #sys.stdout.write('Component mean position: {} +- {}\n'.format(xx[-1],xxE[-1]))
                # Do the plotting
                fig,ax = plt.subplots(figsize=(10,6))

                ax.errorbar(xx,yy,yerr=yyE,xerr=xxE,marker='.',ms=3,ls='none',color='red',elinewidth=0.8)#,label='Comp {}'.format(cid[0]))
                xmin = np.floor(min(xx)-max(xxE))
                #xmin = Rs2mas(100)
                xmax = np.round(max(xx)+max(xxE))
                ymin = np.floor(2*min(yy))
                ymax = np.round(max(yy))
                if yax[jj]=='athin':
                    ymin = -10
                    ymax = -0.5 
                    ax.invert_yaxis()
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)

                ax.set_xscale('symlog',linthresh=np.abs(min(np.abs(xx))),subs=[2, 3, 4, 5, 6, 7, 8, 9])#,linthresh=Rs2mas(100))
                ax.set_yscale('symlog',linthresh=np.abs(min(np.abs(yy))),subs=[2, 3, 4, 5, 6, 7, 8, 9])
                ax.set(xlabel=xlabel, ylabel=ylabel[jj])
                ax.minorticks_on()
                ax.invert_xaxis()
                ax.tick_params(which='both', direction='in')
                #set_corrected_size(fig,figsize)

                fig.subplots_adjust(right=0.8, top=0.98, left=0.06,bottom=0.06)
                outF = '_'.join(outf.split('.')[:-1])+'fit_summary'+yax[jj]+'.pdf'
                fig.savefig(outF,bbox_inches='tight')
                sys.stdout.write('Plot {} has been written to {}\n'.format(yax[jj],outF))
#
    def write_tex_table(self):
        '''Write out a tex table with all modelfit parameters.
        '''
        sys.stdout.write('Writing model component parameters to tex-table.\n')
        i=0
        if os.path.exists('Allmodel.tex'):
            os.remove('Allmodel.tex')
            sys.stdout.write('\nA file Allmodel.tex already exists, will be deleted and created.\n')
        else:
            sys.stdout.write('\nA file Allmodel.tex will be created.\n')

        for i,key in enumerate(self.model):
            if self.shift:
                model = self.model[key].copy()['data_shifted']
            else:
                model = self.model[key].copy()['data']

            model.sort('DIST',reverse=True)
            model['FLUX']*=1e3
            ff = ['%s','%1.0f','%2.2f','%2.2f','%2.2f','%2.1f','%2.2f','%2.2f']
            kk = ['id','FLUX','DELTAX','DELTAY','MAJOR AX','ratio','DIST','logtb']
            formats=dict(zip(kk,ff))
            with open('Allmodel.tex',mode='a') as f:
                f.write('\\midrule\\multicolumn{8}{c}{%2.1f\\,GHz - Band}\\\\\\midrule\n'%self.model[key]['freq'])
#                if i==0:
                model.write(f,format='ascii.no_header', include_names=kk,delimiter='&', formats=formats,fill_values=[(ascii.masked, '  --  ')],overwrite=False)

 #               else:
  #                  dd.write(f,format='ascii.no_header', include_names=kk,delimiter='&', formats=formats,fill_values=[(ascii.masked, '  --  ')],overwrite=False)

            #i+=1

        with open('Allmodel.tex',mode='r') as f:
            file_lines=[''.join([x.strip(),'\\\\', '\n']) for x in f.readlines()]

        freq_list = [str(self.model[key]['freq']) for key in self.keys]
        with open('Allmodel.tex',mode='w') as f:
            f.write('\\begin{table}[!hbt]\n')
            f.write('\\centering\n')
            f.write('\\small\n')
            f.write('\\setlength{\extrarowheight}{1pt}\n')
            f.write('\\setlength{\\tabcolsep}{3pt}\n')
            f.write('\\caption{Parameters of Gaussian model fitting components for '+','.join(freq_list)+'\,GHz observation for '+self.model[self.keys[0]]['source']+'.\label{tab:vlbamodels}\n')
            f.write('\\begin{adjustbox}{width=\linewidth}\n')
            f.write('\\begin{threeparttable}\n')
            f.write('\\begin{tabular}{lSSSSSSc}\\toprule\n')
            f.write('{ID} & {Flux density} & {RA} & {DEC} & {Major \\tnote{1}} & {Ratio\\tnote{2}} & {Distance} & {$\log\,T_\mathrm{b}$} \\\\\n')
            f.write(' & {$[$mJy$]$} & {$[$mas$]$} & {$[$mas$]$} & {$[$mas$]$} & & {$[$mas$]$} & {$[$K$]$} \\\\\n')
#            f.write('\\midrule\n')
            f.writelines(file_lines)
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')
            f.write('\\begin{TableNotes}\n')
            f.write('\\footnotesize\n')
            f.write('\\item[1] FWHM major axis of restoring beam\n')
            f.write('\\item[2] Ratio of minor to major axis\n')
            f.write('\\end{TableNotes}\n')
            f.write('\\end{threeparttable}\n')
            f.write('\\end{adjustbox}\n')
            f.write('\\end{table}\n')
        sys.stdout.write('\nAllmodel.tex has been written.\n')
