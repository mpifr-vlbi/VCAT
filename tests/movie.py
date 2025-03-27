from vcat import ImageCube, ImageData, FitsImage, MultiFitsImage
import matplotlib.pyplot as plt
import glob

#find relevant files
fits_files=glob.glob("../dataset_example/0235+164/**/*.icn.fits",recursive=True)
uvf_files=glob.glob("../dataset_example/0235+164/**/*.uvf",recursive=True)
stokes_q_files=glob.glob("../dataset_example/0235+164/**/*.qcn.fits",recursive=True)
stokes_u_files=glob.glob("../dataset_example/0235+164/**/*.ucn.fits",recursive=True)

#import files
im_cube=ImageCube().import_files(fits_files=fits_files,uvf_files=uvf_files,stokes_q_files=stokes_q_files,stokes_u_files=stokes_u_files)

im_cube.movie(plot_mode="lin_pol",plot_evpa=True,plot_timeline=True,n_frames=100)
