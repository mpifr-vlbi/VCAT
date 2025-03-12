from vcat import ImageData,ImageCube
from vcat.graph_generator import KinematicPlot
import matplotlib.pyplot as plt
import numpy as np

#Import Data
dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        model="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.mfit")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        model="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.mfit")
dataU=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.mfit")
dataW=ImageData("../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.fits",
        model="../dataset_example/3C111_W_2014_05_08/3C111_W_2014_05_08.mfit")
dataX=ImageData("../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.fits",
        model="../dataset_example/3C111_X_2014_05_08/3C111_X_2014_05_08.mfit")


data=[dataC,dataQ,dataU,dataW,dataX]

im_cube=ImageCube(data)

im_cube.plot(overplot_gauss=True,plot_comp_ids=True)

fit=im_cube.fit_comp_spectrum(id=1,plot=True)

#choose the first component collection as a test
print(fit)


"""
#test core-shift with modelcomponent

#first simulate core distances
k_r=1
r0=50
ref_freq=86
dist_0=1

max_i=np.argmax(cc.freqs)

for i in range(len(cc.components)):
    coreshift=r0*((cc.freqs[i]*1e-9/ref_freq)**(-1/k_r)-1)*1e-3
    cc.dist[i]=cc.dist[max_i]-coreshift
    cc.components[i].distance_to_core=cc.components[max_i].distance_to_core-coreshift/cc.scale
"""


#now calculate core shift
fit_result=im_cube.fit_coreshift(id=1,plot=True)
