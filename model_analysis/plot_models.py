'''
This is an example script on how you could use the modelComps class.
It reads in here modelfits and clean maps from three maps 
It also reads in a ascii file containing shifts for these maps, looking as
# FREQS DEC RA
vlba_K_U -0.451 0.298
vlba_Q_K -0.076 -0.152
vlba_Q 0.000 0.000
W_map 0.0 0.0

This file can be created using the alignment module.

'''

from VLBIana.model_analysis.modelComps import *

#Getting lists of the modelfiles and clean files and sorting them after frequency, in this case
modFs   = glob ('modelfits/*.fits')
clFs    = glob('cleanfits/*.fits')
bands   = ['U','K','Q','W']
modFs   = [m for f in bands for m in modFs if m.split('/')[-1].split('_')[1].find(f)!=-1]
clFs    = [m for f in bands for m in clFs if m.split('/')[-1].split('_')[1].find(f)!=-1]

# Loading everything and creating the modelclass. Please provide a redshift, if not, no Brightness Temperature will be calculated.
mods = modelComp(modFs,cleanFiles=clFs,shift='masked_shifts_gmva.txt',z=0.005)

# This is an example on how to change the ids once you got an idea on which one is which. 
# The change_id function takes a list of the old id names and the new id names and changes these in the class itself.
changemod=[]
changemod.extend(['A13','B2','A11','A14','B1','C2','A7','A10','B3','C1','A8','A9','A12'])
changemod.extend(['A13','A14','B2','B3','A10','A11','B1','C2','C1','A7','A8','A9','A12','A15'])
changemod.extend(['A14','A13','B4','A15','B2','A10','A9','B1','EJ','EJ','C2','B3','WJ-B','A12','A11','A7'])
changemod.extend(['A15','B4','A12','A13','B3','A14'])
keys=mods.keys

old_ids=[]
old_ids.extend([str(a)+'_'+str(b) for a,b in zip(mods.model[keys[0]]['data']['ep'],mods.model[keys[0]]['data']['id'])])
old_ids.extend([str(a)+'_'+str(b) for a,b in zip(mods.model[keys[1]]['data']['ep'],mods.model[keys[1]]['data']['id'])])
old_ids.extend([str(a)+'_'+str(b) for a,b in zip(mods.model[keys[2]]['data']['ep'],mods.model[keys[2]]['data']['id'])])
old_ids.extend([str(a)+'_'+str(b) for a,b in zip(mods.model[keys[3]]['data']['ep'],mods.model[keys[2]]['data']['id'])])

mods.change_id(old_ids,changemod)

# To make a plot for components comps=['A13','A15','B2','B3','A14'] and connecting them with a line with X=FREQ and Y=FLUX
mods.plot_comp_xy(comps=['A13','A15','B2','B3','A14'],line=True)

# To overplot the models onto the clean maps giving the dimensions of the image.
mods.overplot_model(ra=10,dec=7)
# To write out a Tex-table for for all component values.
mods.write_tex_table()
