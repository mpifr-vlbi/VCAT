from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt
import numpy as np

dataU=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.uvf",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.modelfits",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")

dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.uvf",
        model="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.modelfits",
        difmap_path="/usr/local/difmap/uvf_difmap_2.5g/")


#test model component alignment
comp1=dataU.components
comps2=dataQ.components


#find largest component in both images
majs1=[]
for comp in comp1:
    majs1.append(comp.maj)

majs2=[]
for comp in comps2:
    majs2.append(comp.maj)

argmax1=np.argmax(majs1)
argmax2=np.argmax(majs2)

#assign the largest component the component number 1
dataU.components[argmax1].component_number=1
dataQ.components[argmax2].component_number=1

#align the images on component with id 1
print("Aligning on Model Components")
dataU=dataU.align(dataQ,method="modelcomp",comp_ids=1)
print("Done.")

im_cube=ImageCube([dataU,dataQ])
print(im_cube)

#plot it
im_cube.plot(overplot_gauss=True)

