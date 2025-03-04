from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt

dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.uvf")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.uvf")

images=[dataC,dataQ]

im_cube=ImageCube(images)

#plot it
im_cube.plot()

#restore it
im_cube_new=im_cube.restore(mode="epoch")

#plot it again
im_cube_new.plot()



