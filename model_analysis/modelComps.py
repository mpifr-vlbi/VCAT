#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.table import Table,vstack
from astropy.io import fits,ascii
from astropy.time import Time
from glob import glob
import sys,os
from itertools import cycle,chain,compress
from VCAT.modules.jet_calculus import *
import VCAT.modules.fit_functions as ff
from VCAT.modules.plot_functions import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

plt.ion()

def trim_axs(axs, N):
  """little helper to massage the axs list to have correct length..."""
  axs = axs.flat
  for ax in axs[N:]:
    ax.remove()
  return axs[:N]
def keyfunc(s):
  return [int(''.join(g)) if k else ''.join(g) for k, g in groupby(s, str.isdigit)]

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
    def __init__(self,modFiles,cleanFiles=None,shift=None,colormap=None,z=None):
        sys.stdout.write('initiate modelComp class.\n')
        self.modFiles =modFiles
        self.model  =   dict()
        self.model_sorted = None
        self.cchead = dict()
        self.ccmap  = dict()
        self.keys   = []
        self.ids    = []
        self.eps    = [chr(value) for value in range(ord('a'),ord('a')+len(self.modFiles))]
        self.z      = z

        if self.z==None:
            sys.stdout.write('Please provide a redshift if Tb should be calculated, giving the parameter of z when calling the modelComp class.\n')
        else:
            sys.stdout.write('Redshift provided, hence Tb will be calculated.\n')

        if colormap:
            self.colormap = colormap
        else:
            self.colormap = 'inferno'
        self.id_colors = []
        self.symbols = []

        for ii,modFile in enumerate(modFiles):
            with fits.open(modFile) as f:
                h = f[0].header
                key = h['OBJECT']+'.'+str(h['CRVAL3']/1e9)+'.'+h['DATE-OBS']
                self.keys.append(key)
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
            self.model[key]['data']['ep'] = nid*[self.eps[ii]]
            self.model[key]['data']['cid'] = mpl.colors.to_hex('red')
#include tb here
            if self.z:
                self.model[key]['data']['tb'] = derive_tb(self.model[key]['data']['FLUX'],self.model[key]['freq'],self.z,self.model[key]['data']['MAJOR AX'],self.model[key]['data']['ratio'])
                self.model[key]['data']['logtb'] = np.log10(self.model[key]['data']['tb'])

    #generate a list of the all components
        [self.ids.extend(self.model[key]['data']['id']) for key in self.keys]
        self.ids = np.unique(self.ids)
        # generate a standard color and symbol for each id
        self.colormap = colormap
        NUM_COLORS = len(self.ids)
        cmap = cm.get_cmap(colormap,NUM_COLORS)
        for i in range(cmap.N):
            self.id_colors.append(mpl.colors.rgb2hex(cmap(i)))
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
                    img = f[0].data
                    if img.ndim == 4:
                        img = img.reshape(img.shape[2],img.shape[3])

                #key = h['OBJECT']+'.'+str(h['CRVAL3']/1e9)+'.'+h['DATE-OBS']
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

    def update_ids(self):
        sys.stdout.write('Updatend component id list.\n')
        self.ids = []
        [self.ids.extend(self.model[key]['data']['id']) for key in self.keys]
        self.ids = np.unique(self.ids)


    def update_cm(self,colormap=False):
        sys.stdout.write('Updating colormap.\n')
        if colormap==False:
            colormap=self.colormap
        else:
            self.colormap = colormap
        self.update_ids()
        NUM_COLORS = len(self.ids)
        cmap = cm.get_cmap(colormap,NUM_COLORS)
        self.id_colors=[]
        for i in range(cmap.N):
            self.id_colors.append(mpl.colors.rgb2hex(cmap(i)))
        self.id_colors = np.array(self.id_colors)
        self.symbols = np.array(['o','v','*','P','s','D'] * int(np.ceil(NUM_COLORS/6)))
        for key in self.keys:
            for ii,model in enumerate(self.model[key]['data']):
                mask = self.ids==model['id']
                model['cid'] = self.id_colors[mask][0]

    def change_id(self,old_ids,new_ids):
        '''
        Usage:
            old_ids = ['a_2','b_5','c_6'] is a list of the component ids to be changed. Consisting of the epoch id (ep) and the component id in that epoch (id)
            new_ids = ['a2','a2','b'] a list of names you want to assign to the components. The id of the component in this epoch will then be changed to this value
        '''
        sys.stdout.write('Change model ids according to imput.\n')
        for ii,oids in enumerate(old_ids):
            epoch,cid = oids.split('_')
            key=self.keys[np.where(np.array(self.eps)==epoch)[0][0]]
            jj=np.where(self.model[key]['data']['id']==cid)[0][0]
            self.model[key]['data']['id'][jj]=new_ids[ii]

        self.update_cm()

    def sort_by_id(self):
        sys.stdout.write('Creating class.model_sorted with components sorted by id.\n')
        self.udate_ids()

        self.model_sorted = dict()
        for ID in self.ids:
            date = []
            for i,key in enumerate(self.model):
                if i ==0:
                    self.model_sorted[ID] = Table(dtype=self.model[key]['data'])
                row = self.model[key]['data'][self.model[key]['data']['id']==ID]
                if len(row) > 0:
                    self.model_sorted[ID].add_row(row[0])
                    date.append(self.model[key]['date'])
            self.model_sorted[ID]['date']= date
        self.update_cm()

###
    def plot_comp_xy(self,xax='DIST',yax='FLUX',out=False,comps=False,line=False,ccolor=False):
        sys.stdout.write('Plot component {} vs {}\n'.format(xax,yax))
        if not self.model_sorted:
            self.sort_by_id()
        if comps:
            model_sorted = {key: self.model_sorted[key] for key in comps}
        else:
            model_sorted = self.model_sorted.copy()

        if ccolor:
            ccolors= len(self.ids)*[ccolor]
        else:
            ccolors = self.id_colors

        fig,ax = plt.subplots(figsize=(12,8))
        for i,comp in enumerate(model_sorted):
            color = ccolors[self.ids==comp['id'][0]]
            symbol = self.symbols[i]
            xx = self.model_sorted[comp][xax]
            yy = self.model_sorted[comp][yax]

            if line:
                ax.plot(xx,yy,marker=symbol, color = color, label = comp,lw=1)
            else:
                ax.scatter(xx,yy,marker=symbol, color = color, label = comp)

        ax.set(xlabel=xax, ylabel=yax)
        fig.legend(loc=7, fontsize='small')
        fig.tight_layout()
        fig.subplots_adjust(right=0.9, top=0.98)

        if out:
            if type(out)==bool:
                outf = self.model[comp]['source']+'_components_all.pdf'
            fig.savefig(outf,bbox_inches='tight')
            sys.stdout.write('Plot has been written to {}'.format(outf))
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

            if out:
                if type(out)==bool:
                    outf = self.model[modF]['source']+'_components.pdf'
                fig.savefig(outf,bbox_inches='tight')
                sys.stdout.write('Plot has been written to {}\n'.format(outf))
            else:
              plt.show()
            plt.cla()

    def overplot_model(self,sigma=3,fig_size='screen',ra=False,dec=False,saveFile=False,plot_color=False,cntr_color=False,cntr_lw=False,ccolor=False,plot_all=True,out=True,plot_id=True,plot_below=False):
        sys.stdout.write('Plot modelcomponents over clean maps\n')
        if plot_all:
            nn = len(self.model)
            if nn>6:
                nx = 4
            if plot_below:
                nx = 1
            else:
                nx = 2
            ny = int(np.ceil(nn/nx))
        else:
            nx = 1
            ny = 1

        if plot_below:
            fig,axs = plt.subplots(ny,nx, figsize=(15,10))
        else:
            fig,axs = plt.subplots(ny,nx, figsize=(12,8))
        axs   = trim_axs(axs,len(self.model))

        if not cntr_color:
                cntr_color = 'grey'
        if not cntr_lw:
            cntr_lw=1

        if ccolor:
            sys.stdout.write('Set user component color\n')
            ccolors= len(self.ids)*[ccolor]
        else:
            ccolors = self.id_colors


        ####################
        # setting all parameters for plotting a clean image
        #####################
        for ax,key in zip(axs,self.keys):
            modelh= self.model[key]
            model = self.model[key]['data']
            clean = self.cchead[key]
            ccmap =  self.ccmap[key]
            if not ra:
                ra  = clean['fov']/3.
                dec = clean['fov']/5.
            if type(ra)==list:
                Dra     = ra
                Ddec = dec
            else:
                Dra = [-ra,ra]
                Ddec= [-dec,dec]

            scale   = -clean['px_inc']*3.6e6
            sigma   = sigma
            level0  = clean['noise']*sigma
            lev=[]
            for i in range(0,10):
                lev.append(level0*2**i)
            xx  = np.linspace(-clean['naxis']*0.5*scale,(clean['naxis']*0.5-1)*scale,clean['naxis'])
            yy  = np.linspace(clean['naxis']*0.5*scale,-(clean['naxis']*0.5-1)*scale,clean['naxis'])
            vmax=0.5*ma.amax(ccmap)
            norm = mpl.colors.SymLogNorm(linthresh=level0,linscale=0.5,vmin=level0,vmax=vmax,base=np.e)

            ##################
            # Plotting
            ###################
            #f,ax = plt.subplots()
            ax.axis('scaled')
            ax.set_xlim(Dra)
            ax.set_ylim(Ddec)
            ax.invert_xaxis()
            plotBeam(clean['beam'][1],clean['beam'][0],clean['beam'][2],ra,-dec+0.2,ax)

            cntr=ax.contour(xx,yy,ccmap,linewidths=cntr_lw,levels=lev,colors=cntr_color,alpha=1)
            if plot_color:
                extent = np.max(xx),np.min(xx),np.min(yy),np.max(yy)
                im = ax.imshow(ccmap,cmap=colormap,extent=extent,origin='lower', interpolation='gaussian')
                im.set_norm(norm)

            Mx  = model['DELTAX']
            My  = model['DELTAY']
            Mposa   = model['POSANGLE']
            Mmaj    = model['MAJOR AX']
            Mmin    = model['MINOR AX']
            for j,xx in enumerate(Mx):
                epid=model['id'][j]
                ccolor = ccolors[self.ids==epid][0]
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
                if maj1_y==maj2_y:
                    ax.plot(maj1_x,maj1_y, color = ccolor,marker='+', markersize=5,lw=1)
                else:
                    ax.plot([maj1_x,maj2_x],[maj1_y,maj2_y], color = ccolor, lw = 1)
                    ax.plot([min1_x,min2_x],[min1_y,min2_y], color = ccolor, lw = 1)
                ax.annotate('{}'.format(model['id'][j]),xy=(Mx[j],My[j]), xycoords='data', xytext=(-5, (-1)**j*40), textcoords='offset points',arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color=ccolor,fontsize=8)

            # set axis, labels, etc.
            ax.set(xlabel='RA [mas]', ylabel='DEC [mas]')
            ax.minorticks_on()
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.annotate('{} - {:.1f} GHz - {}'.format(modelh['source'],modelh['freq'],modelh['date_obs']),xy=(0.03,0.9),xycoords='axes fraction',size=12)

        plt.tight_layout(pad=0.2,w_pad=0.2,h_pad=0.2)
#        fig.subplots_adjust(right=0.9, top=0.98)

        if out:
            if type(out)==bool:
                outf = modelh['source']+'_Model_overplot.pdf'
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

            ff = ['%s','%1.3f','%2.2f','%2.2f','%2.2f','%2.1f','%2.2f','%2.2f']
            kk = ['id','FLUX','DELTAX','DELTAY','MAJOR AX','ratio','DIST','logtb']
            formats=dict(zip(kk,ff))
            with open('Allmodel.tex',mode='a') as f:
                f.write('\\midrule\\multicolumn{8}{c}{%2.1f\\GHz - Band}\\\\\\midrule\n'%self.model[key]['freq'])
#                if i==0:
                self.model[key]['data'].write(f,format='ascii.no_header', include_names=kk,delimiter='&', formats=formats,fill_values=[(ascii.masked, '  --  ')],overwrite=False)

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
            f.write(' & {$[$Jy$]$} & {$[$mas$]$} & {$[$mas$]$} & {$[$mas$]$} & & {$[$mas$]$} & {$[$K$]$} \\\\\n')
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
