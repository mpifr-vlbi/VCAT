from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt
import glob

useDIFMAP=True

#find relevant files
fits_files=glob.glob("../dataset_example/3C111*2014_05_08/**/*.fits",recursive=True)
uvf_files=glob.glob("../dataset_example/3C111*2014_05_08/**/*.uvf",recursive=True)
stokes_q_files=glob.glob("../dataset_example/3C111*2014_05_08/**/*.qcln",recursive=True)
stokes_u_files=glob.glob("../dataset_example/3C111*2014_05_08/**/*.ucln",recursive=True)

print(len(stokes_q_files),len(stokes_u_files),len(uvf_files),len(fits_files))

#import data
im_cube=ImageCube().import_files(fits_files=fits_files,uvf_files=uvf_files,stokes_q_files=stokes_q_files,stokes_u_files=stokes_u_files)

#plot it
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

im_cube=im_cube.align(mode="epoch",useDIFMAP=useDIFMAP)
print(im_cube)
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

#get turnover map
turnover_cube=im_cube.get_turnover_map(specific_pixel=(512,512),limit_freq=False)

#create plots
turnover_cube.plot(plot_mode="turnover",do_colorbar=True,contour=True)
turnover_cube.plot(plot_mode="turnover_flux",do_colorbar=True,contour=True)
turnover_cube.plot(plot_mode="turnover_error",do_colorbar=True,contour=True)
turnover_cube.plot(plot_mode="turnover_chisquare",do_colorbar=True,contour=True)
