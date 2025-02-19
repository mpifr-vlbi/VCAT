from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt

dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.modelfits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.uvf")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.modelfits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.modelfits")

images=[dataC,dataQ]

im_cube=ImageCube(images)

print(im_cube)


#plot

MultiFitsImage(im_cube)
plt.show()
