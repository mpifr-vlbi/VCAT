from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt
import glob

useDIFMAP=True

#find relevant files
fits_files=glob.glob("../dataset_example/0235+164/**/*.icn.fits",recursive=True)
uvf_files=glob.glob("../dataset_example/0235+164/**/*.uvf",recursive=True)
stokes_q_files=glob.glob("../dataset_example/0235+164/**/*.qcn.fits",recursive=True)
stokes_u_files=glob.glob("../dataset_example/0235+164/**/*.ucn.fits",recursive=True)

#import data
im_cube=ImageCube().import_files(fits_files=fits_files,uvf_files=uvf_files,stokes_q_files=stokes_q_files,stokes_u_files=stokes_u_files)

#plot it
im_cube.plot(plot_mode="lin_pol",plot_evpa=True,shared_colormap="epoch",do_colorbar=True)

im_cube=im_cube.align(mode="epoch",useDIFMAP=useDIFMAP)
print(im_cube)
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

spix_cube=im_cube.get_spectral_index_map(freq1=15,freq2=23)
spix_cube.plot(plot_mode="spix",im_colormap=True,contour=True,contour_color="black",do_colorbar=True)

rm_cube=im_cube.get_rm_map(freq1=15,freq2=23)
rm_cube.plot(plot_mode="rm",do_colorbar=True,contour=True,contour_color="black")


im_cube=im_cube.stack(mode="freq")
im_cube.plot(plot_mode="lin_pol",plot_evpa=True)

