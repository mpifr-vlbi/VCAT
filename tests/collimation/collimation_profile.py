#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.io import ascii
from astropy.table import Table,vstack
import os, sys
from glob import glob
from itertools import cycle
from cycler import cycler
from matplotlib.pyplot import cm
import matplotlib as mpl

### from original Plot set: https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/
def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

def set_size(width, fraction=1,subplots=(1,1),ratio=False):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
       Width in pts, or string of predined document type
    fraction: float,optional
       Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
    The number of rows and columns of subplots

    Returns
    -------
    fig_dim: tuple
       Dimensions of figure in inches
    """
    if width.find('_')!=-1:
        w = width.split('_')
        width = w[0]
        fraction= float(w[1])
    if width =='aanda':
        width_pt = 256.0748
    elif width =='aanda*':
        width_pt = 523.5307
    elif width == 'beamer':
        width_pt = 342
    elif width == 'screen':
        width_pt = 600
    else:
        width_pt = width
    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2.
    if not ratio:
        ratio = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio* (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

### for defining colors
""" for VCAT we might think about using standard colors/linestyles and giving the
user the option to provide a different color palette. At the moment I allow this by
the keywords in the plot function to tweak the appearance:
    color=color_palette,
    fit_color=['black','black'],
    fit_ls =['-','-'],
    fit_label=['combined','combined']

Also the ridgeline need to already be shifted for the fitting to make sense.
For this not being true, we could also add the option to provide a shift.
"""
n=8
colors = cmap(np.linspace(0,0.95,n))
#colors = cycler(color=color)
markers = ['x','>','<','+','d','*','p','o','2']
ls = ['-', '--', ':', '-.','-', '--', ':', '-.']
#markers = cycler(mm)

#### I add here the functions for fitting, I suggest to move them to the helper script, if not already there


## End of fitting functions

def axesWidthPlot (ax, **kwargs):
    args ={'xlabel':r'Distance from core \small{[mas]}','ylabel':r'De-convolved width \small{[mas]}','xscale':'log','yscale':'log'}
    args.update(kwargs)

    xlabel = args['xlabel']
    ylabel = args['ylabel']

    ax.set_xscale(args['xscale'])
    ax.set_yscale(args['yscale'])
    if args['xlabel']:
        ax.set_xlabel(xlabel)
    if args['ylabel']:
        ax.set_ylabel(ylabel)
    if 'secxax' in args:
        secx = ax.secondary_xaxis('top', functions=(mas2Rs, Rs2mas))
        secx.set_xlabel(args['secxax'])
    if 'secyax' in args:
        secy = ax.secondary_yaxis('right', functions=(mas2Rs, Rs2mas))
        secy.set_ylabel(args['secyax'])

def plot_fit(x,fitfunc,beta,betaerr,chi2,ax=None,**kwargs):
    args = {'color':'k', 'annotate':False,'asize':8,'annox':0.6,'annoy':0.05,'lw':1,'ls':'-','label':None}
    args.update(kwargs)
    ax = ax or plt.gca()
    if fitfunc == 'scatter':
        function = scatter(beta,x)
        text = '$theta_s={:.2f}\\pm{:.2f}$\n $theta_i={:.2f}\\pm{:.2f}$\n$\\chi_\\mathrm{{red}}^2={:.2f}$'.format(beta[0],betaerr[1],beta[1],betaerr[1],chi2)

    if fitfunc == 'Powerlaw':
        function = powerlaw(beta,x)
        text = '{}\n$k={:.2f}\\pm{:.2f}$\n $\\chi_\\mathrm{{red}}^2={:.2f}$'.format(args['label'],beta[1],betaerr[1],chi2)
    elif fitfunc == 'brokenPowerlaw':
        function = broken_powerlaw(beta,x)
        text = '{}\n$W_\\mathrm{{0}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{u}}={:.2f}\\pm{:.2f}$\n$k_\\mathrm{{d}}={:.2f}\\pm{:.2f}$\n$z_\\mathrm{{B}} = {:.1f}\\pm {:.1f}$ mas'.format(args['label'],beta[0],betaerr[0],beta[1],betaerr[1],beta[2],betaerr[2],beta[-1],betaerr[-1])

    if args['label']:
        ax.plot(x,function,color=args['color'],lw=args['lw'],ls=args['ls'],label=args['label'],zorder=1)
    else:
        ax.plot(x,function,color=args['color'],lw=args['lw'],ls=args['ls'],zorder=1)
    if args['annotate']:
        ax.annotate(text, xy=(args['annox'],args['annoy']),xycoords='axes fraction',size=args['asize'],horizontalalignment='left',verticalalignment='bottom',bbox=bbox_props)

def fit_width(width,dist,
                width_err=False,
                dist_err=False,
                fit_type='brokenPowerlaw',
                x0_bpl=[0.3,0,1,2],
                x0_pl=[0.1,1],
                save_fit = True):
    '''Fit a power-law or broken-powerlaw to jet width'''

    if fit_type == 'Powerlaw':
        if dist_err:
            beta,sd_beta,chi2,out = fit_pl(dist,width,width_err,sx=dist_err,x0=x0_pl)
        else:
            beta,sd_beta,chi2,out = fit_pl(dist,width,width_err,x0=x0_pl)

    elif fit_type=='brokenPowerlaw':
        if dist_err:
            beta,sd_beta,chi2,out = fit_bpl(dist,width,width_err,sx=dist_err,x0=x0_bpl)
        else:
            beta,sd_beta,chi2,out = fit_bpl(dist,width,width_err,x0=x0_bpl)

    return beta, sd_beta, chi2, out
#########################################################

class Jet(object):
    '''Class containing the jet width
    '''
    def __init__(self,
                 inp_data,
                 jet = 'Jet',
                 label = '',
                 shift = None,
                 date = '',
                 freq = '',
                 data_4th_colum = 'dist_err' #if 4th column provided, specify what it is
                ):
        '''Read in jet width data as dictionary with the possible keys:
            'dist':[]
            'width':[]
            'width_err':[]
            'freq':[]
            'date':str/timedate object
            ('peak':[]) optional
            ('peak_err':[]) optional
            data = ascii table with dist,width, width_err,(dist_err),freq
                for data_type='table'
            data = {'mapFile':str, 'ridgeLine': str, 'logFile':str,'label':str,
                'errFile': False, 'cjet':True, 'theta':0, 'shift':False}
                for data_type = 'RL-Laura'

        args:
            jet: 'Jet' or 'Cjet' or 'Twin' for jet or counter jet
            data_4th_column: 'dist_err' or 'freq' if a 4th column is provided in the input table
                please specify what it is
            label : '' the label for this jet type, is also used as label for plotting

        '''
        self.jet = jet
        self.date = date
        self.data = None
        self.label = label
        data = Table.read(inp_data, format='ascii')
        data.rename_column(data.colnames[0],"dist")
        data.rename_column(data.colnames[1],"width")
        data.rename_column(data.colnames[2],"width_err")

        if len(data.columns) == 4:
            if data_4th_colum == 'freq':
                data.rename_column(data.colnames[3],'freq')
            elif data_4th_colum  == 'dist_err':
                data.rename_column(data.colnames[3],'dist_err')
            else:
                sys.stdout.write('Please provide the data type for the 4th column.\n')
        elif len(data.columns) > 4:
            sys.stdout.write('Too many columns provided. Maximum is four.\n')

        data.add_column(date, name='date')
        data.add_column(label, name='label')

        if not 'freq' in data.colnames:
            try:
                data.add_column(freq,name='freq')
            except:
                sys.stdout.write("No frequency given in data file, please provide the observing frequency as freq=float.\n")
                sys.exit(0)
        for cols in ['dist','width','width_err','freq']:
            data[cols].info.format = '{:.2f}'

        if jet == 'Jet':
            data['dist'] = np.abs(data['dist'])
            data.add_column('jet',name='jet')
            self.data = data
        elif jet == 'CJet':
            data['dist'] = np.abs(data['dist'])
            sys.stdout.write('Cjet type given, so column gets identifier cjet.\n')
            data.add_column('cjet',name='jet')
            self.data = data
        elif jet == 'Twin':
            dataj,datacj = [data[data['dist']>0],data[data['dist']<0]]
            datacj['dist'] = np.abs(datacj['dist'])
            dataj.add_column('jet',name='jet')
            datacj.add_column('cjet',name='jet')
            self.data = vstack([dataj,datacj])
        else:
            sys.stdout.write('Please provide a lable for the jet, is it "Jet", "CJet", or "Twin"?\n')


class JetWidthCollection(object):
    '''A collection of jet width epochs
    '''
    def __init__(self,
                 Jet_list=[], #list of Jet objects
                 label = ['']
                ):
        self.Jets = Jet_list
        self.label = label
        # Variable initializing for fittings
        self.beta = []
        self.sd_beta = []
        self.chi2 = []
        self.out = []
        self.fit_label = []
        self.fit_type = []
        self.fit_data = []
        self.filter_data = []

        for i,j in enumerate(self.Jets):
            if label[0] == '':
                j['label'] = str(i)

        self.jet_data = [j.data for j in self.Jets]
        self.data = vstack(self.jet_data)

    def filter_jet(self,filter_by):
        '''Filter the jet data Table.
        '''
        filter_data = self.data
        try:
            filter_data = filter_data[filter_data['jet'] == filter_by['jet']]
            sys.stdout.write('filter out jet = {}\n'.format(filter_by['jet']))
        except:
            pass
        try:
            filter_data = filter_data[filter_data['date'] < filter_by['date_max']]
        except:
            pass
        try:
            filter_data = filter_data[filter_data['date'] == filter_by['date']]
            sys.stdout.write('filter out date = {}\n'.format(filter_by['date']))
        except:
            pass
        try:
            filter_data = self.data[filter_data['date'] > filter_by['date_min']]
        except:
            pass
        try:
            filter_data = filter_data[filter_data['freq'] < filter_by['freq_max']]
        except:
            pass
        try:
            filter_data = filter_data[filter_data['freq'] > filter_by['freq_min']]
        except:
            pass
        try:
            filter_data = filter_data[filter_data['label'] == filter_by['label']]
        except:
            pass
        self.filter_data = filter_data
        return filter_data

    def fitJet(self, filter_by = False, label = '', fit_type = 'brokenPowerlaw', **kwargs):
        '''Calls the fit function defined above after selectingt the data to be fitted.
        It is possible to do several fits and save them for plotting.
        Args:
            filter_by: {'jet':'jet','date_max':XX,'date_min','freq_max':XX,'freq_min','label':XX} basically anything can be given
        '''
        if filter_by:
            fit_data =self.filter_jet(filter_by=filter_by)
        sys.stdout.write('Data to be fitted:\n{}\n Now will run fitting.\n'.format(fit_data))

        width = fit_data['width']
        width_err = fit_data['width_err']
        dist = fit_data['dist']
        try:
            dist_err = fit_data['dist_err']
            beta, sd_beta, chi2, out = fit_width(width,dist,width_err,dist_err=dist_err,fit_type=fit_type,x0_bpl=[0.3,0,1,2],x0_pl=[0.1,1])
        except:
            beta, sd_beta, chi2, out = fit_width(width,dist,width_err,fit_type=fit_type,x0_bpl=[0.3,0,1,2],x0_pl=[0.1,1])

        self.fit_type.append(fit_type)
        self.fit_label.append(label)
        self.fit_data.append(fit_data)
        self.beta.append(beta)
        self.sd_beta.append(sd_beta)
        self.chi2.append(chi2)
        self.out.append(out)


##############################################################
    def plotCollimation(self,
                        jet = '', #Can be 'Jet','Cjet','Twin', or '' (all data will be used in last case)
                        filter_by = False,
                        plot_fit_result = False,
                        plot_fit_data = False,
                        flux_uncertainty = 0.15,
                        write_fit_info=True,
                        add_data=False,
                        fig_extension="pdf",
                        plot_line=False,
                        fig_size = 'aanda*',
                        saveFile = 'Plot_collimation',
                        label = [],
                        color = colors,
                        marker = markers,
                        fit_color = 'k',
                        fit_ls = '-',
                        fit_label = False,
                        asize = 8
                       ):
        '''Plotting the fit over the width
        Args:
            data : [width,width_err,distance(, distance_err)]
        '''
        if plot_fit_result:
            saveFile += '_fit'
            if type(plot_fit_result) == dict:
                sys.stdout.write('Use provided fit function\n')
                fit_type = plot_fit_result['fit_type']
                beta = plot_fit_result['beta']
                sd_beta = plot_fit_result['sd_beta']
                chi2 = plot_fit_result['chi2']
                fit_jet = plot_fit_result['fit_jet']
            else:
                try:
                    fit_type,beta,sd_beta,chi2,fit_jet = [],[],[],[],[]
                    index = 0
                    if plot_fit_result:
                        sys.stdout.write("Will plot collimation fits: {}.\n".format(plot_fit_result))
                        index = [self.fit_label.index(p) for p in plot_fit_result]
                        try:
                            for i in index:
                                fit_type.append(self.fit_type[i])
                                beta.append(self.beta[i])
                                sd_beta.append(self.sd_beta[i])
                                chi2.append(self.chi2[i])
                                fit_jet.append(self.fit_data[i])
                        except:
                            sys.stdout.write("There are multiple fits saved in this class. Please specify the fit_name with plot_fit_result.\n")
                            sys.exit(0)
                    elif len(self.fit_type)==1:
                        print('Only one fit in CollimationCollection, will plot this one.\n')
                        fit_type.append(self.fit_type[0])
                        beta.append(self.beta[0])
                        sd_beta.append(self.sd_beta[0])
                        chi2.append(self.chi2[0])
                        fit_jet.append(self.fit_data[0])
                except:
                    sys.stdout.write('Something went wrong with selecting the fitting data.\n')
                    sys.exit(0)

        if plot_fit_data:
            data = self.fit_data
        elif filter_by:
            data = [self.filter_jet(filter_by)]
        else:
            data = self.jet_data

        data_by_jet, data_j, data_cj = [],[],[]
        if jet == 'Twin':
            for i,d in enumerate(data):
                data_by_jet.append(data[i].group_by('jet'))
                data_j.append(data_by_jet[i].groups[data_by_jet[i].groups.keys['jet'] == 'jet'])
                data_cj.append(data_by_jet[i].groups[data_by_jet[i].groups.keys['jet'] == 'cjet'])
        elif jet == 'Jet':
            for i,d in enumerate(data):
                data_by_jet.append(data[i].group_by('jet'))
                data_j.append(data_by_jet[i].groups[data_by_jet[i].groups.keys['jet'] == 'jet'])
        elif jet =='Cjet':
            for i,d in enumerate(data):
                data_by_jet.append(data[i].group_by('jet'))
                data_j.append(data_by_jet[i].groups[data_by_jet[i].groups.keys['jet'] == 'cjet'])
        else:
            sys.stdout.write('Use all data assuming a one-sided jet.\n')
            data_j = data

        if len(label) == 0:
            sys.stdout.write('Provide a label for "jet" if you want another label to be used during plotting.\n')
            label = self.label

        nsub=1

        ymin = 6e-4
        ymax = max(self.data['width'])*3
        xmax = max (np.abs(self.data['dist']))*1.1
        xmin = Rs2mas(100)
        xr=np.arange(xmin,xmax,0.01)
        yrmin = -1
        yrmax = 1

        if jet == 'Twin':
            figsize=(set_size(fig_size,subplots=(nsub,2)))
            f,ax=plt.subplots(nsub,2,sharex='col',sharey='row',gridspec_kw={'hspace': 0, 'wspace': 0},figsize=figsize)
            axes = ax.flatten()

        else:
            figsize=(set_size(fig_size,subplots=(nsub,1)))
            f,ax=plt.subplots(nsub,1,sharex='col',sharey='row',gridspec_kw={'hspace': 0, 'wspace': 0},figsize=figsize)
            axes = [ax]


    ########### Now do the plotting ######

        i=0
        if jet == 'Twin':
            sys.stdout.write('Plot Twin jet.\n')
            for jdata,cjdata in zip (data_j,data_cj):
                mm = marker[i]
                cc = color[i]
                axes[0].scatter(jdata['dist'],jdata['width'],s=4,marker=mm,color=cc,label='{}'.format(label[i]))
                axes[0].errorbar(jdata['dist'],jdata['width'],yerr=jdata['width_err'],fmt=mm,ms=0,color=cc,linewidth=0,elinewidth=0.4,errorevery=1,label='_nolegend_',alpha=0.3)
                axes[1].scatter(cjdata['dist'],cjdata['width'],s=4,marker=mm,color=cc)
                axes[1].errorbar(cjdata['dist'],cjdata['width'],yerr=cjdata['width_err'],fmt=mm,ms=0,color=cc,linewidth=0,elinewidth=0.4,errorevery=1,label='_nolegend_',alpha=0.3)
                i+=1
        else:
            sys.stdout.write('Plot only one jet.\n')
            plot_data = data_j
            for i,jdata in enumerate(plot_data):
                mm = marker[i]
                cc = color[i]
                ax.scatter(jdata['dist'],jdata['width'],s=4,marker=mm,color=cc,label='{}'.format(label[i]))
                ax.errorbar(jdata['dist'],jdata['width'],yerr=jdata['width_err'],fmt=mm,ms=0,color=cc,linewidth=0,elinewidth=0.4,errorevery=1,label='_nolegend_',alpha=0.3)

        if plot_fit_result:
            kj,kcj = 0,0
            annoxj,annoxcj = 0.1,0.6
            for i,jj in enumerate(fit_jet):
                mm = markers[i]
                if fit_color:
                    cc = fit_color[i]
                else:
                    cc = 'k'
                if fit_ls:
                    ls = fit_ls[i]
                else:
                    ls = '-'
                if fit_label:
                    label = fit_label[i]
                else:
                    label = fit_type[i]
                if any(np.unique(jj['jet']) == 'jet'):
                    sys.stdout.write('Plot fit jet.\n')
                    if kj>0:
                        annoxj = 0.1+0.5*kj
                    if kj>1:
                        write_fit_info = False
                    plot_fit(xr,fit_type[i],beta[i],sd_beta[i],chi2[i],ax=axes[0],annotate=write_fit_info,asize=asize,ls=ls,lw='0.8',label=label, color=cc,annox=annoxj,annoy=0.05)
                    kj += 1
                elif any(np.unique(jj['jet']) == 'cjet'):
                    sys.stdout.write('Plot fit counter jet.\n')
                    if kcj>0:
                        annoxcj = 0.6-0.5*kcj
                    if kcj>1:
                        write_fit_info = False
                    plot_fit(xr,fit_type[i],beta[i],sd_beta[i],chi2[i],ax=axes[1],annotate=write_fit_info,asize=asize,ls=ls,lw='0.8',label=label, color=cc,annox=annoxcj,annoy=0.05)
                    kcj += 1
                else:
                    sys.stdout.write('Somethings wrong, not plotting fit.\n')

    ### finalize axis settings
        handles, labels = axes[0].get_legend_handles_labels()


        if jet == 'Twin':
            legend_col = 5
            axesWidthPlot(axes[0],secxax=r'Distance from core [$R_\mathrm{S}$]')
            axesWidthPlot(axes[1],secxax=r'Distance from core [$R_\mathrm{S}$]',secyax=r'De-convolved width \small{[$R_\mathrm{S}$]}',ylabel=False)
            axes[0].annotate('Eastern Jet', xy=(0.2,0.85),xycoords='axes fraction',size=14)
            axes[1].annotate('Western Jet', xy=(0.2,0.85),xycoords='axes fraction',size=14)
            axes[0].legend(handles,labels,loc='lower left',ncol=legend_col,markerscale=1,labelspacing=0.1,bbox_to_anchor=(0.0, 1.3,2.,.4),mode='expand',borderaxespad=0.,handletextpad=0.1) #original axes[0]
            for axs in ax.flatten():
                axs.set_xscale('log')
                axs.set_yscale('log')
                axs.set_xlim(xmin,xmax)
                axs.set_ylim(ymin,ymax)
                axs.minorticks_on()
                axs.tick_params(which='both',direction='inout')
                axs.label_outer()

            axes[0].invert_xaxis()
        else:
            legend_col=2
            axes[0].legend(handles,labels,loc='lower left',ncol=legend_col,markerscale=1,labelspacing=0.1,bbox_to_anchor=(0.0, 1.15,2.,.4),borderaxespad=0.,handletextpad=0.1) #original axes[0]
            axesWidthPlot(ax,secxax=r'Distance from core [$R_\mathrm{S}$]')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.minorticks_on()
            ax.tick_params(which='both',direction='inout')
            ax.label_outer()

        saveFile = saveFile+'.'+fig_extension
        plt.savefig(saveFile,bbox_inches='tight',transparent=True)
        plt.close()
        sys.stdout.write('Saved plot to file: {}\n'.format(saveFile))
