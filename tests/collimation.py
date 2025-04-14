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

#Test collimation profile for image cube
plot=im_cube.fit_collimation_profile(method="model",jet="Twin",label="Model",color="black")

im_cube.fit_collimation_profile(method="ridgeline",jet="Twin",label="Ridgeline",color="green",show=True)

