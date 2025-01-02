from VLBIana.align_imagesEHTim import *
from glob import glob

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

label = ['Freq_low', 'Freq_high']
maps = ['Freq_low.fits', 'Freq_high.fits']

header = 'FREQS DEC RA'

'''
beam can be set to "mean", then a mean beamsize will be computed. Otherwise give the beamsize in mas
offset negative means left, angle as defined from east(left) clockwise
'''
rr = plot_aligned_maps(maps=maps,
                       beam='max',
                       e_maj=100,
                       e_min=100,
                       e_pa=90,
                       e_xoffset=-100,
                       do_err=True    # this enables the calculation and plotting
                       # of uncertainties for the spectral index
                       )

shift = []
shift_freqs = []
maps2_shifted = []
inc_dec, inc_ra = [],[]
masked = True
shift.append(rr['shift'])
shift_freqs.append(label[0]+'_'+label[1])
maps2_shifted.append(rr['file2'])
inc_dec.append(rr['increment_dec'])
inc_ra.append(rr['increment_ra'])

# sys.stdout.write('\n Final shifts {}\n'.format(shift))

data = np.zeros(len(shift_freqs),dtype=[('fq','U6'),('dec',float),('ra',float)])
data['fq'] = np.array(shift_freqs)
data['dec']	= np.array([sr[0] for sr in shift])
data['ra'] = np.array([sr[1] for sr in shift])
np.savetxt('shifts.txt', data, fmt='%s %.3f %.3f', delimiter='\t', header=header)
