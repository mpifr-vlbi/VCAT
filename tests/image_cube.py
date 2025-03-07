from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt

dataC=ImageData("../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_C_2014_05_08/3C111_C_2014_05_08.uvf")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.uvf")

images=[dataC,dataQ]

im_cube=ImageCube(images)

#plot it
im_cube.plot(mode="epoch",xlim=[10,-10])

#regrid it
im_cube=im_cube.regrid(mode="all",npix=1024,pixel_size=0.1,useDIFMAP=True)
print("Regridded to 1024,0.1")
im_cube.plot()

#restore it
im_cube=im_cube.restore(useDIFMAP=True)
print("Restored with common beam")
im_cube.plot()

#shift it
im_cube=im_cube.shift(shift_x=10,shift_y=10,mode="freq",useDIFMAP=True)
print("shifted by 5,5")
im_cube.plot()



