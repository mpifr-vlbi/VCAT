import numpy as np

def rms_est(x_arr, thresh=5):
    # Flatten the input array
    x_flat = x_arr.ravel()
    # Calculate the root mean square of the flattened array
    rms_in = np.sqrt(np.nanmean(x_flat**2))
    
    # Create a mask where values are either NaN or greater than the threshold times the initial RMS
    mask_rms = np.isnan(x_arr) | (x_arr > thresh * rms_in)
    
    # Extract squared values of elements where the mask is False (i.e., elements within the threshold)
    maps_value = x_arr[~mask_rms]**2
    
    # Calculate the RMS based on the extracted squared values
    rms = np.sqrt(np.nanmean(maps_value))
    
    # While the RMS changes significantly, update the mask and recalculating the RMS
    while not np.isclose(rms, rms_in):
        rms_in = rms
        mask_rms = np.isnan(x_arr) | (x_arr > thresh * rms_in)
        maps_value = x_arr[~mask_rms]**2
        rms = np.sqrt(np.nanmean(maps_value))
        
    # Create the final map removing the NaNs values  
    mask = x_arr > thresh * rms 

    # Return the final RMS and mask
    return rms, mask
