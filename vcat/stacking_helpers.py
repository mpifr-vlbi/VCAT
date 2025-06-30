#this script can stack VLBI .fits images in stokes parameters and in linear polarization and EVPA (weighted and unweighted)
#it can also do some beam folding operations required prior to stacking

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.modeling import models, fitting
from matplotlib.collections import LineCollection  
import pexpect
from pexpect import replwrap  
import os
from numpy import linalg
from tqdm import tqdm
from scipy.optimize import minimize

#initialize logger
from vcat.config import logger
from vcat.helpers import write_mod_file_from_components,getComponentInfo


def stack_images(image_array, #input images to be stacked
        weighted=False, #choose whether to use the weighted option
        weights_array=[] #put in weights to use for stacking
        ):
    
    #make sure everything is numpy
    for ind,image in enumerate(image_array):
        image_array[ind]=np.array(image)

    for ind,weights in enumerate(weights_array):
        weights_array[ind]=np.array(weights)
    
    #get some general properties
    dim=image_array[0].shape
    stacked_image=np.empty(shape=dim)
    n_images=len(image_array)
    
    #check if all input images have the same dimension
    wrong_size=False
    for image in image_array:
        if image.shape != dim:
            wrong_size=True
    
    #check that weights_array has same dimension as the image_array
    wrong_weights=False
    if weighted:
        for weights in weights_array:
            if weights.shape != dim:
                wrong_weights = True
        if len(weights_array) != n_images:
            wrong_weights = True 
    else:
        #setup a weights_array with equal weights if weighting is not used
        weights_array=np.empty((n_images,dim[0],dim[1]))
        for i in range(n_images):
            weights_array[i]=np.ones(dim)


    if wrong_size:
        raise Exception("Error! Your input arrays do not all have the same dimension!")
    elif wrong_weights:
        raise Exception("Error! You have selected a weighted stack but your weights are not the same dimension as your images!")
    else:
        weight_sum=np.zeros_like(image_array[0])
        for k in tqdm(range(n_images)):
            stacked_image+=weights_array[k]*image_array[k]
            weight_sum+=weights_array[k]
        stacked_image=stacked_image/weight_sum
        
        return stacked_image

#this function takes file paths to fits files (e.g. from DIFMAP or CASA exportfits) and creates a stacked fits file from them
#by stacking I,Q,U and all IFs independent from each other. With the "align" option, the images will be centered at the brightest pixel in total intensity (aligned)
#it can be either passed fits_files including all polarizations or

def stack_fits(fits_files, #a list of filepaths to fits files (either full polarization or just stokes I)
        stokes_q_fits=[], #a list of filepaths to fits files containing stokes Q
        stokes_u_fits=[], #a list of filepaths to fits files conataining stokes U
        export_fits=True, #choose whether to write an output fits file (stacked)
        output_file="stacked.fits", #choose file name for output file
        overwrite=True, #choose whether to overwrite an already existing image or not
        align=False #choose whether to align the images on the brightest pixel in Stokes I before stacking (all pols will be aligned according to the brightest pixel in Stokes I)
        ):
    
    #check if there is more than one fits file
    wrong_len=False
    if len(fits_files)<1:
        wrong_len=True
    
    #check if fits files are in STOKES format
    not_stokes=True
    for fits_file in fits_files:
        file=fits.open(fits_file)
        for i in range(10):
            try:
                if "STOKES" in file[0].header["CTYPE"+str(i)]:
                    not_stokes=False
            except:
                pass
    
    #check if FITS file contains more than just Stokes I
    only_stokes_i=False
    for fits_file in fits_files:
        file=fits.open(fits_file)
        if file[0].data.shape[0]==1:
            only_stokes_i=True

    if wrong_len:
        raise Exception("Error! Please put in more than one fits file, otherwise stacking makes no sense!")
    elif not_stokes:
        raise Exception("Error! Your fits-files are not in STOKES format. This is currently not implemented!")
    else:
        if only_stokes_i and (len(stokes_q_fits)!=len(fits_files) or len(stokes_u_fits)!=len(fits_files)):
            logger.warning("Warning! Only Stokes I input given!")
            logger.warning("-> will produce only Stokes I stacked image")
            pols=1 #do only stokes I in this case
        else:
            pols=3 #do all three polarization Stokes I, Q, U

        dim=fits.open(fits_files[0])[0].data.shape
        output_stacked=np.empty((pols,dim[1],dim[2],dim[3]))

        for pol in range(pols): #iterate over polarizations
            for spw in range(dim[1]): #iterate over spws/IFs
                data_to_stack=np.empty((len(fits_files),dim[2],dim[3]))
                if only_stokes_i:
                    if pol==0:
                        for ind,fits_file in enumerate(fits_files):
                            data_to_stack[ind]=fits.open(fits_file)[0].data[0][spw]
                    if pol==1:
                        for ind,fits_file in enumerate(stokes_q_fits):
                            data_to_stack[ind]=fits.open(fits_file)[0].data[0][spw]
                    if pol==2:
                        for ind,fits_file in enumerate(stokes_u_fits):
                            data_to_stack[ind]=fits.open(fits_file)[0].data[0][spw]
                else:
                    for ind,fits_file in enumerate(fits_files):
                        data_to_stack[ind]=fits.open(fits_file)[0].data[pol][spw]
                
                #store stokes i data for later use
                if pol==0:
                    i_data_to_stack=data_to_stack

                #align polarizations according to stokes i peak
                if align:
                    logger.info(f"Stacking Polarization {pol}.")
                    output_stacked[pol][spw]=stack_images(align_images(data_to_stack,align_by=i_data_to_stack))
                else:
                    logger.info(f"Stacking Polarization {pol}.")
                    output_stacked[pol][spw]=stack_images(data_to_stack)

        if export_fits:
            file=fits.open(fits_files[0])
            file[0].data=output_stacked
            file[0].header["DATE-OBS"]="3000-01-01T00:00:00.000000" #we will write an arbitrary year into the header
            file[1].header['XTENSION'] = 'BINTABLE'
            file.writeto(output_file,overwrite=overwrite)
        
        return output_stacked

#this functioon takes file paths to fits files (e.g. from DIFMAP) and creates a stacked fits file from them
#by first calculation linear polarization P and EVPA and stacking P and EVPA and NOT Q,U. If weighted is set to true,
#the EVPA stack is weighted with the linear Polarization. The "align" option centers all polarizations on the Stokes I peak before stacking

def stack_pol_fits(fits_files, #list of file paths to fits files with full polarization or Stokes I only data
        stokes_q_fits=[], #list of file paths to fits files with Stokes Q data
        stokes_u_fits=[], #list of file paths to fits files with Stokes U data
        weighted=False, #choose whether to weight the EVPA stacking by the level of linear polarization
        align=False #choose whether to align the images on the brightest pixel in Stokes I before stacking (all pols will be aligned according to the brightes pixel in Stokes I)
        ):

    #check if there is more than one fits file
    wrong_len=False
    if len(fits_files)<1:
        wrong_len=True
    
    #check if fits files are in STOKES format
    not_stokes=True
    for fits_file in fits_files:
        file=fits.open(fits_file)
        for i in range(10):
            try:
                if "STOKES" in file[0].header["CTYPE"+str(i)]:
                    not_stokes=False
            except:
                pass

    #check if FITS file contains more than just Stokes I
    only_stokes_i=False
    for fits_file in fits_files:
        file=fits.open(fits_file)
        if file[0].data.shape[0]==1:
            only_stokes_i=True

    if wrong_len:
        raise Exception("Error! Please put in more than one fits file, otherwise stacking makes no sense!")
    elif not_stokes:
        raise Exception("Error! Your fits-files are not in STOKES format. This is currently not implemented!")
    elif only_stokes_i and (len(stokes_q_fits)!=len(fits_files) or len(stokes_u_fits)!=len(fits_files)):
        raise Exception("Error! At least one of your input files contains only Stokes I but no polarization information and you did not specify a matching Stokes Q and U path!")
    else:
    
        dim=fits.open(fits_files[0])[0].data.shape
        output_stacked=np.empty((3,dim[1],dim[2],dim[3]))

        #read in all fits files
        image_storage=np.empty((len(fits_files),3,dim[1],dim[2],dim[3]))
        for ind,fits_file in enumerate(fits_files):
           
            if only_stokes_i: #if the fits files only contain one polarization, it reads from the different input files
                image_storage[ind]=[fits.open(fits_file)[0].data[0],fits.open(stokes_q_fits[ind])[0].data[0],fits.open(stokes_u_fits[ind])[0].data[0]]
            else: #if the fits file contains multiple polarization it will read from the single fits file
                image_storage[ind]=fits.open(fits_file)[0].data[0:3]     

        #calculate linear polarization and EVPAs from Stokes Q and U for all input images and IFs/spw
        pol_images=np.empty((dim[1],3,len(fits_files),dim[2],dim[3])) #-> here we will only save stokes i,lin.pol and evpa
        for ind,image in enumerate(image_storage):
            for spw in range(dim[1]):
                stokes_i=image[0][spw]
                stokes_q=image[1][spw]
                stokes_u=image[2][spw]
                pol_images[spw][0][ind] = stokes_i #Stokes I
                pol_images[spw][1][ind] = np.sqrt(stokes_q**2+stokes_u**2) #linear polarization
                pol_images[spw][2][ind] = 0.5*np.arctan2(stokes_u,stokes_q) #EVPA
    
        #do the actual stacking
        for spw in range(dim[1]):
            
            if align:
                output_stacked[0][spw]=stack_images(align_images(pol_images[spw][0])) #stack stokes_i
                output_stacked[1][spw]=stack_images(align_images(pol_images[spw][1],align_by=pol_images[spw][0])) #stack lin.pol.
                if weighted:
                    output_stacked[2][spw]=stack_images(align_images(pol_images[spw][2],align_by=pol_images[spw][0]),weighted=True,weights_array=pol_images[spw][1])    
                else:
                    output_stacked[2][spw]=stack_images(align_images(pol_images[spw][2],align_by=pol_images[spw][0])) #stack EVPA without weights

            else:
                output_stacked[0][spw]=stack_images(pol_images[spw][0]) #stack stokes_i
                output_stacked[1][spw]=stack_images(pol_images[spw][1]) #stack lin.pol.
                if weighted:
                    output_stacked[2][spw]=stack_images(pol_images[spw][2],weighted=True,weights_array=pol_images[spw][1])    
                else:
                    output_stacked[2][spw]=stack_images(pol_images[spw][2]) #stack EVPA without weights

        return output_stacked

#this method folds images with a defined beam (either custom input or common beam), needs .uvf,.mod and .fits files from the original images as inputs
#requires the definition of "difmap_path", since it runs difmap in the background, so you need to point difmap_path to where your difmap executable sits

def fold_with_beam(fits_files, #array of file paths to fits images input
        difmap_path, #path to where the difmap executable is located
        bmaj=-1, #beam major axis to fold with (in mas)
        bmin=-1, #beam minor axis to fold with (in mas)
        posa=-1, #beam position angle to fold with (in deg)
        shift_x=0, #additional shift in x direction (in mas)
        shift_y=0, #additional_shift in y direction (in mas)
        channel="i", #polarization channel to use for folding (default stokes "i"), possible options "q","u","v"
        output_dir="output", #Name and path to the output directory
        outname="fits_convolved", #outname of the file
        n_pixel=2048, #number of pixels in output image
        pixel_size=-1, #pixel size in mas (default uses 1/10 of bmin)
        use_common_beam=False, #choose whether to fold with the common beam of the input arrays (TRUE) or whether to use the input bmaj,bmin,posa beam (FALSE, default)  
        mod_files=[], #optional input array of file paths to .mod files for fits_files
        clean_mod_files=[], #optional input array of file paths to the clean .mod files for aligning the uvf file first with selfcal
        uvf_files=[], #optional input array of file paths to .uvf files for fits_files
        weighting=[0,-1], #weighting option for uvw in DIFMAP, default is natural weighting
        uvtaper=[1,0],
        do_selfcal=True,
        ):

        if use_common_beam:
            bmaj,bmin,posa=get_common_beam(fits_files) 
        
        #set pixel_size if not manually set:
        if pixel_size<0:
            pixel_size=bmin/10

        #check if there was input to mod_files
        if len(fits_files)!=len(mod_files):
            mod_files=[]
            logger.info("No or insufficient number of mod files defined. Will try to guess their names from .fits file names")
            for fits_file in fits_files:
                mod_files=np.append(mod_files,fits_file[:-5]+".mod")
        
        #check if there was input to uvf_files
        if len(fits_files)!=len(uvf_files):
            uvf_files=[]
            logger.info("No or insufficient number of uvf files defined. Will try to guess their names from .fits file names")
            for fits_file in fits_files:
                uvf_files=np.append(uvf_files,fits_file[:-5]+".uvf")

        env = os.environ.copy()

        # add difmap to PATH
        if difmap_path != None and not difmap_path in os.environ['PATH']:
            env['PATH'] = env['PATH'] + ':{0}'.format(difmap_path)

        # remove potential difmap boot files (we don't need them)
        env["DIFMAP_LOGIN"] = ""
        # Initialize difmap call
        child = pexpect.spawn('difmap', encoding='utf-8', echo=False, env=env)
        child.expect_exact("0>",None, 2)

        def send_difmap_command(command,prompt="0>"):
            child.sendline(command)
            child.expect_exact(prompt, None, 2)
            logger.debug(command)
            logger.debug("DIFMAP Output: %s", child.before)

        for ind, fits_file in enumerate(fits_files):
            send_difmap_command("obs " + uvf_files[ind])

            send_difmap_command(f"uvw {weighting[0]},{weighting[1]}")  #use custom weighting
            if uvtaper!=[1,0]:
                send_difmap_command(f"uvtaper {uvtaper[0]},{uvtaper[1]}")
            if abs(shift_x)>0 or abs(shift_y)>0:
                send_difmap_command(f"shift {shift_x},{shift_y}")
            if do_selfcal:
                send_difmap_command("select i")
                send_difmap_command("rmod " + clean_mod_files[ind])
                send_difmap_command("selfcal")
            send_difmap_command("select " + channel)
            send_difmap_command("rmod " + mod_files[ind])
            send_difmap_command("maps " + str(n_pixel) + "," + str(pixel_size))
            if not (bmaj==-1 and bmin ==-1 and posa==-1):
                send_difmap_command("restore " + str(bmaj) + "," + str(bmin) + "," + str(posa))
            send_difmap_command("save " + outname)

        os.system("rm -rf difmap.log*")

def modelfit_difmap(uvf_file,mod_file,niter,difmap_path,components="",weighting=[0,-1],channel="i",do_selfcal=False,selfcal_model=""):

    env = os.environ.copy()

    # add difmap to PATH
    if difmap_path != None and not difmap_path in os.environ['PATH']:
        env['PATH'] = env['PATH'] + ':{0}'.format(difmap_path)

    # remove potential difmap boot files (we don't need them)
    env["DIFMAP_LOGIN"] = ""
    # Initialize difmap call
    child = pexpect.spawn('difmap', encoding='utf-8', echo=False, env=env)
    child.expect_exact("0>", None, 2)

    def send_difmap_command(command, prompt="0>"):
        child.sendline(command)
        child.expect_exact(prompt, None, 2)
        logger.debug(command)
        logger.debug("DIFMAP Output: %s", child.before)

    #do the modelfit in DIFMAP
    send_difmap_command("obs " + uvf_file)
    send_difmap_command(f"uvw {weighting[0]},{weighting[1]}")  # use custom weighting
    if do_selfcal:
        send_difmap_command("select i")
        send_difmap_command("rmod " + selfcal_model)
        send_difmap_command("selfcal")
    send_difmap_command("select " + channel)
    send_difmap_command("rmod " + mod_file)
    send_difmap_command("modelfit  "+str(niter))
    send_difmap_command("wmod " + mod_file)

    os.system("rm -rf difmap.log*")

    if components!="":
        #get fitted components from .mod file
        model_df = getComponentInfo(mod_file)

        for ind, comp in model_df.reset_index().iterrows():
            # assign new values
            components[ind].x=comp["Delta_x"]
            components[ind].y=comp["Delta_y"]
            components[ind].maj=comp["Major_axis"]
            components[ind].min=comp["Minor_axis"]
            components[ind].pos=comp["PA"]
            components[ind].flux=comp["Flux"]

        return components
    else:
        logger.warning("No components provided will only write modfile")
        return None

def modelfit_ehtim_pol(uvf_file,components,niter,npix=1024,
                       nwalker=200,minimizer="dynesty_dynamic",fov=10,plot=False,
                       max_size=5,max_flux=1,max_dist=5,
                       circ_gauss=False,export_model="",skip_fit=False):
    """
    code to modelfit gaussian components to polarization (no Stokes-I) using ehtim
    Args:
        uvf_file: .uvf file
        components: Starting model as component objects
        niter: number of iterations
        npix: number of  pixels for ehtim to consider
        nwalker: number of walkers in bayesian fitting
        fov: field of view for ehtim to consider in mas
        max_size: Maximum component size in mas
        max_flux: Maximum component flux in Jy
        max_dist: Maximum distance from center in mas
        circ_gauss (bool): If True, will only fit circular gaussians (Radius = maj!)
        plot: decide whether to create model plot

    Returns:

    """
    import ehtim as eh

    #fov in mas!!
    scale=components[0].scale

    try:
        npix=int(npix)
    except:
        npix=1024
    try:
        fov=float(fov)
    except:
        fov=10

    obs=eh.obsdata.load_uvfits(uvf_file)

    mod = eh.model.Model()

    params = []
    for comp in components:
        params.append(comp.lin_pol)
        params.append(comp.maj * scale)
        params.append(comp.min * scale)
        params.append(comp.pos)
        params.append(comp.x * scale)
        params.append(comp.y * scale)
        params.append(comp.evpa)

    params = np.array(params).reshape(-1, 7)

    for param in params:
        # actually add it
        if circ_gauss:
            mod = mod.add_circ_gauss(
                F0=param[0],
                FWHM=param[1]*eh.RADPERUAS,
                x0=param[4] / scale / 180 * np.pi,
                y0=param[5] / scale / 180 * np.pi,
                pol_frac=1,
                pol_evpa=param[6] / 180 * np.pi
            )
        else:
            mod = mod.add_gauss(
                F0=param[0],
                FWHM_maj=param[1] *eh.RADPERUAS*1e3,
                FWHM_min=param[2] *eh.RADPERUAS*1e3,
                PA=param[3] / 180 * np.pi,
                x0=param[4] / scale / 180 * np.pi,
                y0=param[5] / scale / 180 * np.pi,
                pol_frac=1,
                pol_evpa=param[6] / 180 * np.pi)

    # make image from model
    im = mod.make_image(fov*1e3*eh.RADPERUAS, npix)

    #setup prior for modelling
    mod_prior = mod.default_prior(fit_pol=True)

    for i,param in enumerate(params):
        mod_prior[i]["F0"] = {'prior_type': 'flat', 'min': 0, 'max': max_flux}
        mod_prior[i]["x0"] = {'prior_type': 'flat', 'min': -max_dist * 1e3 * eh.RADPERUAS,
                              'max': max_dist * 1e3 * eh.RADPERUAS}
        mod_prior[i]["y0"] = {'prior_type': 'flat', 'min': -max_dist * 1e3 * eh.RADPERUAS,
                              'max': max_dist * 1e3 * eh.RADPERUAS}
        mod_prior[i]["pol_frac"] = {'prior_type': 'fixed'}
        mod_prior[i]["pol_evpa"] = {'prior_type': 'flat', 'min': 0, 'max': np.pi}

        if circ_gauss:
            mod_prior[i]["FWHM"] = {'prior_type': 'flat', 'min': 0.01 * 1e3 * eh.RADPERUAS,
                                        'max': max_size * 1e3 * eh.RADPERUAS}
        else:
            mod_prior[i]["FWHM_maj"] = {'prior_type':'flat', 'min':0.01*1e3*eh.RADPERUAS, 'max':max_size*1e3*eh.RADPERUAS}
            mod_prior[i]["FWHM_min"] = {'prior_type': 'flat', 'min': 0.01 * 1e3 * eh.RADPERUAS,
                                        'max': max_size * 1e3 * eh.RADPERUAS}
            mod_prior[i]["PA"] = {'prior_type':'flat','min':0, 'max': np.pi}

    if not skip_fit:
        if minimizer=="dynesty_dynamic":
            run_nested_kwargs = {'maxiter': niter}
            kwargs = {'run_nested_kwargs': run_nested_kwargs}
            minimizer_kwargs = {"nlive": nwalker,'sample':'rslice'}
            mod_fit = eh.modeler_func(obs, mod, mod_prior, d1='pvis', minimizer_func=minimizer, fit_pol=True,
                                      alpha_d1=20, minimizer_kwargs=minimizer_kwargs, pol1='I', pol2='Q',
                                      pol3='U', processes=0, **kwargs)
        else:
            minimizer_kwargs = {'options':{'maxiter':niter}}
            mod_fit = eh.modeler_func(obs, mod, mod_prior, d1='pvis', minimizer_func=minimizer, fit_pol=True,
                                      alpha_d1=20, minimizer_kwargs=minimizer_kwargs, pol1='I', pol2='Q',
                                      pol3='U', processes=0)
        # plot final model
        if plot:
            im = im.blur_gauss(obs.fit_beam(weighting="natural"))
            im.display(plotp=True, show=True, export_pdf="fitted_model.pdf")

        if export_model != "":
            mod_fit["model"].save_txt(export_model)

        for ind, params in enumerate(mod_fit["model"].params):
            if circ_gauss:
                components[ind].lin_pol = params["F0"]
                components[ind].maj = params["FWHM"] / np.pi * 180
                components[ind].min = params["FWHM"] / np.pi * 180
                components[ind].pos = 0
                components[ind].x = params["x0"] / np.pi * 180
                components[ind].y = params["y0"] / np.pi * 180
                components[ind].evpa = params["pol_evpa"] / np.pi * 180
            else:
                components[ind].lin_pol = params["F0"]
                components[ind].maj = params["FWHM_maj"] / np.pi * 180
                components[ind].min = params["FWHM_min"] / np.pi * 180
                components[ind].pos = params["PA"] / np.pi * 180
                components[ind].x = params["x0"] / np.pi * 180
                components[ind].y = params["y0"] / np.pi * 180
                components[ind].evpa = params["pol_evpa"] / np.pi * 180

        return components
    else:
        if export_model != "":
            mod.save_txt(export_model)
        return components



def modelfit_ehtim_old(uvf_file,components,niter,npix=1024,fov=10):
    import ehtim as eh

    #fov in mas!!
    scale=components[0].scale

    try:
        npix=int(npix)
    except:
        npix=1024
    try:
        fov=float(fov)
    except:
        fov=10

    obs=eh.obsdata.load_uvfits(uvf_file)

    def get_chisq(params, obs, plot=False):
        # create empty model
        mod = eh.model.Model()

        params = np.array(params).reshape(-1, 7)

        for param in params:
            # actually add it
            mod = mod.add_gauss(F0=param[0],
                                FWHM_maj=param[1] *eh.RADPERUAS*1e3,
                                FWHM_min=param[2] *eh.RADPERUAS*1e3,
                                PA=param[3] / 180 * np.pi,
                                x0=param[4] / scale / 180 * np.pi,
                                y0=param[5] / scale / 180 * np.pi,
                                pol_frac=1,
                                pol_evpa=param[6] / 180 * np.pi)

        # make image from model
        im = mod.make_image(fov*1e3*eh.RADPERUAS, npix)

        if plot:
            im = im.blur_gauss(obs.fit_beam(weighting="natural"))
            im.display(plotp=True, show=True, export_pdf="fitted_model.pdf")

        # calculate reduced chi-squared (still need to make this fully correct!!!)
        chisq_red = (obs.chisq(im, pol="Q") + obs.chisq(im, pol="U")) / 2

        return chisq_red

    initial_guess=[]
    for comp in components:
        initial_guess.append(comp.lin_pol)
        initial_guess.append(comp.maj*scale)
        initial_guess.append(comp.min*scale)
        initial_guess.append(comp.pos)
        initial_guess.append(comp.x*scale)
        initial_guess.append(comp.y*scale)
        initial_guess.append(comp.evpa)

    progress = []
    def callback(params):
        current_chi2 = get_chisq(params, obs)
        progress.append(current_chi2)
        print(f"Step {len(progress)}: chi2_red = {current_chi2:.4f}, params:{params}")

    callb=get_chisq(initial_guess,obs)
    print(f"Step 0: chi2_red = {callb}")

    result = minimize(get_chisq, initial_guess, args=(obs,False),method="Nelder-Mead",callback=callback,options={"maxiter":niter})

    #plot final model
    fitted_params=result.x
    models = np.array(fitted_params).reshape(-1, 7)

    for ind,model in enumerate(models):
        components[ind].lin_pol=model[0]
        components[ind].maj=model[1]/scale
        components[ind].min=model[2]/scale
        components[ind].pos=model[3]
        components[ind].x=model[4]/scale
        components[ind].y=model[5]/scale
        components[ind].evpa=model[6]

    return components
