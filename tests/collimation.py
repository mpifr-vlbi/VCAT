from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt

dataU=ImageData("../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.uvf",
        model="../dataset_example/3C111_U_2014_05_08/3C111_U_2014_05_08.mfit")
dataQ=ImageData("../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.fits",
        uvf_file="../dataset_example/3C111_Q_2014_05_08/3C111_Q_2014_05_08.uvf")

#dataU=dataU.center()
#dataQ=dataQ.center()

images=[dataU,dataQ]

im_cube=ImageCube(images)

#im_cube.plot()

im_cube.get_ridgeline()

im_cube.plot(plot_ridgeline=True,plot_counter_ridgeline=True)

#Test collimation profile for single image

im1=im_cube.images.flatten()[0]

plot=im1.fit_collimation_profile(method="ridgeline",jet="Twin",label="Ridgeline",color="black",show=True)

#im1.fit_collimation_profile(method="ridgeline",plot=plot,label="Ridgeline",color="blue",show=True)




im_cube.get_ridgeline_profile()

im_cube.plot_evolution()

print(im_cube)

#plot it
im_cube.plot(mode="freq",xlim=[[10,-10],[5,-5]])


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



