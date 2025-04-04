from collimation_profile import *
import os
from glob import glob
import operator

data = ['43GHz-luca.txt','86GHz_coll-9-jan-25.txt','EHT.txt']
freq = [43,86,230]
label = [r'43\,GHz \tiny{(GMVA)}',r'86\,GHz \tiny{(GMVA)}',r'230\,GHz \tiny{(EHT)}']

color_palette = [
#    '#d1d1e0',  # grey for old data 
#    '#b3b3cc',  # grey for old data 
#    '#9494b8',  # grey for old data 
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow
]

jet = []
for i,d in enumerate(data):
    jet.append(Jet(inp_data=data[i], freq = freq[i], label=label[i], jet='Twin',date=2023))

# above jet can also be loaded with jet='Jet' or jet='Cjet'

jet_C = JetWidthCollection(Jet_list=jet, label=label)

jet_C.fitJet(filter_by={'jet':'jet','date':2023},label='jet')
jet_C.fitJet(filter_by={'jet':'cjet','date':2023},label='cjet')
#jet_C.fitJet(filter_by={'jet':'jet','freq':86},label='jet_86GHz')

fit_color = ['black','#d62728']
jet_C.plotCollimation(saveFile='Plot_collimation_jet',plot_fit_result=['jet'],color=color_palette, fit_color=fit_color, fit_ls =['-','--'],fit_label=['Fit_all'])
jet_C.plotCollimation(saveFile='Plot_collimation_jet',plot_fit_result=['jet'],color=color_palette, fit_color=fit_color, fit_ls =['-','--'],fit_label=['Fit_all'],jet='Jet')
jet_C.plotCollimation(saveFile='Plot_collimation_Twin',plot_fit_result=['jet','cjet'],color=color_palette, fit_color=fit_color, fit_ls =['-','-'],fit_label=['Fit_all','Fit_all'],jet='Twin')
