from align_imagesEHTim import *
from glob import glob
########################################################
label = ['L', 'C', 'X']
vlba_maps = ['a.fits', 'b.fits', 'c.fits']  # list containing the fits files of the images. 

###############
'''
the function 'plot_aligned_maps returns file1 as was used for the cross-correlation,file2 as was used for the cross-correlation,shift in mas,increment_dec in mas,increment_ra in mas
in addition several plots are produced.
In the following I produce some lists to write the output of the function to in order to have everything sorted nicely afterwards.
As you see there are some 'if' statements. I just used different masks for different frequency pairs and applyed it that way. You can also just remove the 'if' statement and use the same mask for all.
Possible masks are (numbers given are all in px, probably you have to test a little until you find the right number of pixels):
    cut_left=x pixel from the center of the image to the left and cut everything that is left of this vertical line
    cut_right=x as for left
    e_maj,e_min,e_pa: elliptical mask centered at the map center, with the keyword e_xoffset it can be moved along the x-axis, if needed I could also add the option to move along the y-axis. e_pa is not needed to be given, if left out of the function the major ax is aligned with the y axis
    radius= circular mask
    npix_x,npix_y: rectangular cut again centered at map center
    flux_cut= x : every pixel with a flux smaller than x times the peakflux of the map is removed.
'''
shift, shift_err, diffphase = [], [], []
shift_freqs = []
maps2_shifted = []
inc_dec, inc_ra = [], []
masked = True
for i, files in enumerate(vlba_maps):
    maps = [vlba_maps[i], vlba_maps[i+1]]
    if i == 0:
        rr = plot_aligned_maps(maps, cut_left=50, masked_shift=masked)
    elif i == 1:
        rr = plot_aligned_maps(maps, e_maj=100, e_min=60, e_xoffset=10,
                               masked_shift=masked)
    shift.append(rr['shift'])
    if masked:
        shift_err.append(0.0)
        diffphase.append(0.0)
    else:
        shift_err.append(rr['error'])
        diffphase.append(rr['diffphase'])
    shift_freqs.append(label[i]+'_'+label[i+1])
    maps2_shifted.append(rr['file2'])
    inc_dec.append(rr['increment_dec'])
    inc_ra.append(rr['increment_ra'])

sys.stdout.write('\n Final shifts {}\n'.format(shift))
sys.stdout.write('\n Shift error {}\n'.format(shift_err))

'''
below I write arrays of the shifts relativ to the lowest frequency I had (1GHz) and to the highest frequency (43GHz)
this sets either the lowest frequency map at (0,0) or the highest frequency map.
This should work with any set of images.
'''
shift_ref1GHz_freq  = [label[0], label[0]+'_'+label[1]]
shift_ref1GHz_dec   = [0, shift[0][0]]
shift_ref1GHz_ra    = [0, shift[0][1]]
for i in range(len(shift)-1):
    shift_ref1GHz_freq.append(label[i+1]+'_'+label[i+2])
    shift_ref1GHz_dec.append(shift_ref1GHz_dec[-1]+shift[i+1][0])
    shift_ref1GHz_ra.append(shift_ref1GHz_ra[-1]+shift[i+1][1])

shift_ref43GHz_freq = [label[-1], label[-1]+'_'+label[-2]]
shift_ref43GHz_dec  = [0, -shift[-1][0]]
shift_ref43GHz_ra   = [0, -shift[-1][1]]
for i in range(len(shift)-1):
    shift_ref43GHz_freq.append(label[-2-i]+'_'+label[-3-i])
    shift_ref43GHz_dec.append(shift_ref43GHz_dec[-1]-(shift[-2-i][0]))
    shift_ref43GHz_ra.append(shift_ref43GHz_ra[-1]-(shift[-2-i][1]))

shift_ref43GHz_ra.reverse()
shift_ref43GHz_dec.reverse()
shift_ref43GHz_freq.reverse()

#Now I write the results to txt files, probably you should give other names for the files;)
header = 'FREQS DEC RA'
data = np.zeros(len(shift_ref43GHz_freq),dtype=[('fq', 'U6'), ('dec', float), ('ra', float)])
data['fq']  = np.array(shift_ref43GHz_freq)
data['dec'] = np.array(shift_ref43GHz_dec)
data['ra']  = np.array(shift_ref43GHz_ra)
np.savetxt('shifts_relative_to_43GHz.txt', data, fmt='%s %.3f %.3f', delimiter='\t', header=header) #change file name
#
data['fq']  = np.array(shift_ref1GHz_freq)
data['dec'] = np.array(shift_ref1GHz_dec)
data['ra']  = np.array(shift_ref1GHz_ra)
np.savetxt('shifts_relative_to_1.5GHz.txt', data, fmt='%s %.3f %.3f', delimiter='\t', header=header) #change file name
#
# this writes a txt file with the derived shifts
data = np.zeros(len(shift_freqs), dtype=[('fq', 'U6'), ('dec', float), ('ra', float)])
data['fq']  = np.array(shift_freqs)
data['dec'] = np.array([sr[0] for sr in shift])
data['ra']  = np.array([sr[1] for sr in shift])
np.savetxt('shifts_pairwise.txt', data, fmt='%s %.3f %.3f', delimiter='\t', header=header)
