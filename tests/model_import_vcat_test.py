from vcat import ImageCube
import glob

uvf_files=glob.glob("uvf_files/*.uvf")
model_files=glob.glob("modelfit_files/*.fits")

im_cube=ImageCube().import_files(uvf_files=uvf_files,model_fits_files=model_files)


#read in component association from GUI
im_cube.import_component_association("component_info.csv")

#Do kinematics
print(im_cube.get_speed(1))
