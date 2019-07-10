#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""

todo: test on '/home/dominecf/LIMBA/SEM-dle_data_obsolentni/2019/06cerven/190613_319B_254A-FH/319B-FH-190613'
and '/home/dominecf/LIMBA/SEM-dle_cisla_vzorku/2019/283C_FH-190317_190611/283C_190611'

todo: 
    np.pad(aa, mode='constant', pad_width=[[1,2],[3,4]])
    np.all(aa[1] == 5)

I use scipy.ndimage for image processing.
"""

## User settings

# correlation of images settings
rel_max_shift=.3        ## pixels cropped from the second image determine the maximum shift to be detected (higher number results in slower computation)
decim=2        ## decimation of images prior to correlation (does not affect the results much)
databar_pct = (61./484)   ## relative height of databar at the images' bottom - to be clipped prior to correlation
#rel_smoothing = 15./300   ## smoothing of the correlation map (not the output), relative to image width
rel_smoothing = .01   ## smoothing of the correlation map (not the output), relative to image width
#rel_smoothing = False   ## no smoothing of the correlation map
plot_correlation  = 1 ## diagnostics

# Image post-processing settings
channel_exponent = 1. ## pixelwise exponentiation of image
saturation_enhance = .15
unsharp_weight = 2
unsharp_radius = 50




## Import common moduli
import matplotlib, sys, os, time, collections, imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d, convolve2d
from scipy.ndimage.filters import laplace
from scipy.ndimage import affine_transform
np.warnings.filterwarnings('ignore')

def find_shift(im1, im2, rel_smoothing=rel_smoothing, plot_correlation_name=None):
    """
    shifts im2 against im1 so that a best correlation is found, returns a tuple of the pixel shift
    """
    ## compute the image correlation, and its smoothed Laplacian
    corr = correlate2d(laplace(im1), im2, mode='valid')
    cr=1  # post-laplace cropping, there were some edge artifacts
    if rel_smoothing:
        rel_smoothing_kernel = np.outer(2**-(np.linspace(-2,2,int(im1.shape[0]*rel_smoothing))**2), 2**-(np.linspace(-2,2,int(im1.shape[0]*rel_smoothing))**2))
        rel_smoothing_kernel /= np.sum(rel_smoothing_kernel)
    else:
        rel_smoothing_kernel = [[1]]
    laplcorr = convolve2d(np.abs(corr), rel_smoothing_kernel, mode='valid')[cr:-2-cr,cr:-cr] # simple rel_smoothing and removing spurious last 2lines
    vsize, hsize  = laplcorr.shape

    ## find optimum translation for the best image match
    raw_shifts = (np.unravel_index(np.argmax(laplcorr), laplcorr.shape)) # x,y coords of the minimum in the correlation map
    vshift_rel, hshift_rel = int((vsize/2 - raw_shifts[0] + 0.5)*decim), int((hsize/2 - raw_shifts[1] - 0.5)*decim) # linear transform against image centre
    print('second image is vertically and horizontally shifted by ({:},{:}) px against the previous one'.format(vshift_rel,hshift_rel))

    if plot_correlation_name:
        fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(15,15))
        im = ax.imshow(laplcorr) # fig.colorbar(im, ax=ax, shrink=0.5)
        def plot_cross(h,v,c):
            ax.plot([h/2-5-.5,h/2+5-.5],[v/2+.5,v/2+.5],c=c,lw=.5)
            ax.plot([h/2-.5,h/2-.5],[v/2-5+.5,v/2+5+.5],c=c,lw=.5)
        plot_cross(hsize, vsize, 'k')
        plot_cross(hsize-hshift_rel*2/decim, vsize-vshift_rel*2/decim, 'w')
        fig.savefig(plot_correlation_name, bbox_inches='tight')

    return [vshift_rel, hshift_rel]

def my_affine_tr(im, trmatrix):
    t0 = time.time()
    troffset = np.dot(np.eye(2)-trmatrix, np.array(im_unsharp.shape)/2)
    return affine_transform(im, trmatrix, offset=troffset, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    print('affine tr took', time.time()-t0)
def find_affine(im1, im2, shift_guess, trmatrix_guess, verbose=False):
    """
    optimizes the image overlap through finding not only translation, but also skew/rotation/stretch of the second image 
    """
    crop_up     = int(im2.shape[0]/2-im1.shape[0]/2-shift_guess[0])
    crop_bottom = int(im2.shape[0]/2-im1.shape[0]/2-shift_guess[0])
    crop_left   = int(im2.shape[1]/2-im1.shape[1]/2-shift_guess[1])
    crop_right  = int(im2.shape[1]/2-im1.shape[1]/2-shift_guess[1])
    print(np.sum(im1[]*im2))
    return np.array([[1, .2], [0, 1]])
#def find_shift2(im1, im2, rel_smoothing=rel_smoothing, plot_correlation_name=None):


def paste_overlay(bgimage, fgimage, vs, hs, color, normalize=np.inf):
    #extend_symmetric(composite_output, im.shape[0]+vshift_sum, im.shape[1]+hshift_sum)
    for channel in range(3):
        bgimage[image_padding-vs:image_padding+fgimage.shape[0]-vs, 
                image_padding-hs:image_padding+fgimage.shape[1]-hs, channel] += np.clip(fgimage**channel_exponent*float(color[channel])/normalize**channel_exponent, 0, 1)


## Image manipulation routines
#def extend_canvas():
def safe_imload(imname):
    im = imageio.imread(imname.lstrip('+')) * 1.0  # plus sign has a special meaning of an 'extra' image
    if len(im.shape) > 2: im = im[:,:,0] # using monochrome images only; strip other channels than the first
    return im, (imname[0] == '+' or ('SC' in imname))
def unsharp_mask(im, weight, radius, radius2=None, clip_to_max=True):
    unsharp_kernel = np.outer(2**-(np.linspace(-2,2,radius)**2), 2**-(np.linspace(-2,2,radius2 if radius2 else radius)**2))
    unsharp_kernel /= np.sum(unsharp_kernel)
    if len(np.shape(im)) == 3:
        unsharp = np.dstack([convolve2d(channel, unsharp_kernel, mode='same', boundary='symm') for channel in im])
    else:
        unsharp = convolve2d(im, unsharp_kernel, mode='same', boundary='symm')
    im = np.clip(im*(.7+weight) - unsharp*weight, 0, np.max(im) if clip_to_max else np.inf)
    return im
def saturate(im, saturation_enhance):
    monochr = np.dstack([np.sum(im, axis=2)]*3)
    return np.clip(im*(1.+saturation_enhance) - monochr*saturation_enhance, 0, np.max(im))



colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len(sys.argv)-1))   ## Generate a nice rainbow scale
channel_outputs, channel_names = [], []
extra_outputs, extra_names = [], []
for image_name, color in zip(sys.argv[1:], colors):
    ## Load an image
    im, is_extra = safe_imload(image_name)
    max_shift = int(rel_max_shift*im.shape[0])
    image_padding = max_shift*len(sys.argv[1:]) ## FIXME safe very wide black padding, could be optimized!
    im2crop = im[max_shift:-max_shift-int(im.shape[0]*databar_pct):decim, max_shift:-max_shift:decim]*1.0

    if 'prev_image' in locals():    
        im1crop = prev_image[:-int(im.shape[0]*databar_pct):decim, ::decim]*1.0

        ## Find the best correlation of both images
        t0 = time.time()
        vshift_rel, hshift_rel = find_shift(im1crop, im2crop, plot_correlation_name=image_name.rsplit('.')[0] + '_correlation.png' if plot_correlation else None)
        print('correlation took', time.time()-t0)

        ## Find the optimum affine transform of both images
        trmatrix_sum += find_affine(im1crop, im2crop, trmatrix_sum, verbose=True) - np.eye(2)
    else:
        vshift_rel, hshift_rel, vshift_sum, hshift_sum = 0, 0, 0, 0     ## Initialize position to centre
        trmatrix_sum = np.eye(2) ## Initialize affine transform to identity



    if is_extra:
        extra_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(extra_output, im, vshift_sum+vshift_rel, hshift_sum+hshift_rel, [1,1,1,1], normalize=np.max(im2crop))
        extra_outputs.append(extra_output)
        extra_names.append(image_name)
    else:
        ## Process the new added image
        im_unsharp = unsharp_mask(im, weight=unsharp_weight, radius=unsharp_radius) 
        if 'M21S' in image_name: 
            #vshift_rel, hshift_rel = 0,0     # explicit image locking for "troubled cases"
            im_unsharp = my_affine_tr(im_unsharp, trmatrix_sum)
            
            

        ## Prepare the composite canvas with the first centered image
        if 'composite_output' not in locals(): composite_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(composite_output, im_unsharp, vshift_sum+vshift_rel, hshift_sum+hshift_rel, color, normalize=np.max(im2crop))

        ## Export an individual channel
        channel_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(channel_output, im_unsharp, vshift_sum+vshift_rel, hshift_sum+hshift_rel, color, normalize=np.max(im2crop))
        channel_outputs.append(channel_output)
        channel_names.append(image_name)

        ## Remember the new transform, and store the image as a background for fitting of the next one
        vshift_sum += vshift_rel; hshift_sum += hshift_rel 
        prev_image = im


#imageio.imsave('composite_' + '.png', composite_output)
for n,(i,f) in enumerate(zip(channel_outputs, channel_names)): imageio.imsave('channel' + str(n) + '_'+f.split('.')[0]+'.png', i)
for n,(i,f) in enumerate(zip(extra_outputs, extra_names)): imageio.imsave('extra' + str(n)+ '_'+f.lstrip('+').split('.')[0]+ '.png', i)

    #np.dstack([im2*256/np.max(im2),im1*256/np.max(im1),im2*256/np.max(im2)])

#clipped_sample = composite_output[composite_output.shape[0]//2-100:composite_output.shape[0]//2+100, composite_output.shape[1]//2-100:composite_output.shape[1]//2+100, :]
#composite_output = (composite_output - np.min(clipped_sample)) * np.max(composite_output) / np.max(clipped_sample)  
#imageio.imsave('clipped_sample' + '.png', clipped_sample)
imageio.imsave('composite.png', composite_output)

imageio.imsave('composite_saturate' + '.png', saturate(composite_output, saturation_enhance=saturation_enhance))

