#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""
Multichannel overlay 

Most important features:
    * Automatically aligns images so that their edge features are matched. (Technically, it maximizes abs value of Laplacians.)
    * Optionally, affine transform can be used for better matching images that are rotated or slightly distorted. 
    * Exports the coloured composite image, along with each channel separately. 
    * Additionally, "extra" images can be aligned and exported in grayscale, but they do not affect alignment of further channels. 
      This can be used, e.g., to align SEM images to a series of spectrally resolved cathodoluminescence mappings.

Additional features
    * Images can be sharpened and contrast-enhanced
    * Since electron-microscope images can contain a databar, the bottom portion of the images is ignored during alignment

TODOs: 
    * put image-manip routines into a separate module
    * join the export with annotate.py
    * don't forget to anisotropically scale at the load time
    * (too aggressive clipping, or clipping channels too early?) colors are still not as crisp as they used to be 
    * is there some colour mismatch between affine-tr and normal operation?
    * interactive GUI D&D application
    * test on windows
"""

## User settings

# Settings for correlation of images:
use_affine_transform = 0    ## enables scaling, tilting and rotating the images; otherwise they are just shifted
rel_max_shift=.15           ## pixels cropped from the second image determine the maximum shift to be detected (higher number results in slower computation)
decim=2                     ## decimation of images prior to correlation (does not affect the results much)
databar_pct = (61./484)     ## relative height of databar at the images' bottom - to be clipped prior to correlation
#databar_pct = 0.01     ## relative height of databar at the images' bottom - to be clipped prior to correlation
#rel_smoothing = 15./300    ## smoothing of the correlation map (not the output), relative to image width
rel_smoothing = .005         ## smoothing of the correlation map (not the output), relative to image width
#rel_smoothing = False      ## no smoothing of the correlation map
plot_correlation  = False    ## diagnostics
consecutive_alignment = True ## if disabled, images are aligned always to the first one

EXTRA_IMG_IDENT = 'S'   # each image containing this in its name is treated as extra
EXTRA_IMG_LABEL = '+'   # each image name preceded by this is treated as extra
def is_extra(imname): return (imname[0] == EXTRA_IMG_LABEL or (EXTRA_IMG_IDENT in Path(imname).stem.upper())) ## TODO this should be better defined...

# Image post-processing settings:
channel_exponent = 1. ## pixelwise exponentiation of image (like gamma curve)
saturation_enhance = .15
unsharp_weight = 2 #2
unsharp_radius = 30




## Import common moduli
import matplotlib, sys, os, time, collections, imageio
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d, convolve2d
from scipy.ndimage.filters import laplace, gaussian_filter
from scipy.ndimage import affine_transform, zoom
np.warnings.filterwarnings('ignore')

def find_shift(im1, im2):
    """
    shifts im2 against im1 so that a best correlation is found, returns a tuple of the pixel shift
    """
    corr = correlate2d(laplace(im1), im2, mode='valid')     ## search for best overlap of edges (~ Laplacian of the image correlation)
    cr=1  # post-laplace cropping, there were some edge artifacts
    lc = np.abs(gaussian_filter(corr, sigma=rel_smoothing*im1.shape[1]) if rel_smoothing else corr) 
    raw_shifts = (np.unravel_index(np.argmax(np.abs(lc)), lc.shape)) # x,y coords of the optimum in the correlation map
    vshift_rel, hshift_rel = int((lc.shape[0]/2 - raw_shifts[0] - 0.5)*decim), int((lc.shape[1]/2 - raw_shifts[1] - 0.5)*decim) # centre image

    fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(15,15)); im = ax.imshow(lc)  # 4 lines for diagnostics only:
    def plot_cross(h,v,c): ax.plot([h/2-5-.5,h/2+5-.5],[v/2+.5,v/2+.5],c=c,lw=.5); ax.plot([h/2-.5,h/2-.5],[v/2-5+.5,v/2+5+.5],c=c,lw=.5)
    plot_cross(lc.shape[1], lc.shape[0], 'k'); plot_cross(lc.shape[1]-hshift_rel*2/decim, lc.shape[0]-vshift_rel*2/decim, 'w')
    fig.savefig('correlation_'+image_name.replace('TIF','PNG'), bbox_inches='tight')

    return np.array([vshift_rel, hshift_rel]), np.eye(2) # shift vector (plus identity affine transform matrix)

def my_affine_tr(im, trmatrix, shiftvec=np.zeros(2)): ## convenience function around scipy's implementation
    troffset = np.dot(np.eye(2)-trmatrix, np.array(im.shape)/2) ## transform around centre, not corner 
    if np.all(np.isclose(trmatrix, np.eye(2))): return im
    return affine_transform(im, trmatrix, offset=shiftvec+troffset, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

def find_affine(im1, im2, verbose=False):
    """
    optimizes the image overlap through finding not only translation, but also skew/rotation/stretch of the second image 

    im2 should always be smaller than im1, so that displacement still guarantees 100% overlap
    """
    crop_up     = int(im1.shape[0]/2-im2.shape[0]/2)
    crop_bottom = int(im1.shape[0]/2-im2.shape[0]/2+.5)
    crop_left   = int(im1.shape[1]/2-im2.shape[1]/2)
    crop_right  = int(im1.shape[1]/2-im2.shape[1]/2+.5)
    def fitf(p): 
        return -np.abs(np.sum(laplace(im1[crop_up:-crop_bottom,crop_left:-crop_right])*my_affine_tr(im2, p[2:].reshape(2,2), shiftvec=p[:2])))

    from scipy.optimize import differential_evolution
    m = .1 ## maximum relative affine transformation
    bounds = [(-max_shift,max_shift), (-max_shift,max_shift), (1-m, 1+m), (0-m, 0+m),(0-m, 0+m),(1-m, 1+m)]
    result = differential_evolution(fitf, bounds=bounds)
    return result.x[:2]*decim*.999, result.x[2:].reshape(2,2)

def find_affine_and_shift(im1, im2):
    if use_affine_transform:    return find_affine(im1, im2)    ## Find the optimum affine transform of both images (by fitting 2x2 matrix)
    else:                       return find_shift(im1, im2)     ## Find the best correlation of both images by brute-force search

def paste_overlay(bgimage, fgimage, shiftvec, color, normalize=np.inf):
    for channel in range(3):
        vs, hs = shiftvec.astype(int)
        vc = int(bgimage.shape[0]/2 - fgimage.shape[0]/2)
        hc = int(bgimage.shape[1]/2 - fgimage.shape[1]/2)
        bgimage[vc-vs:vc+fgimage.shape[0]-vs, 
                hc-hs:hc+fgimage.shape[1]-hs, 
                channel] += np.clip(fgimage**channel_exponent*float(color[channel])/normalize, 0, 1)
                #fgimage**channel_exponent*float(color[channel]) 


## Image manipulation routines
def safe_imload(imname):
    im = imageio.imread(str(Path(imname).parent / Path(imname).name.lstrip(EXTRA_IMG_LABEL))) * 1.0  # plus sign has a special meaning of an 'extra' image
    if len(im.shape) > 2: im = im[:,:,0] # using monochrome images only; strip other channels than the first
    return im
def unsharp_mask(im, weight, radius, radius2=None, clip_to_max=True):
    unsharp_kernel = np.outer(2**-(np.linspace(-2,2,radius)**2), 2**-(np.linspace(-2,2,radius2 if radius2 else radius)**2))
    unsharp_kernel /= np.sum(unsharp_kernel)
    if len(np.shape(im)) == 3:      # handle channels of colour image separately
        unsharp = np.dstack([convolve2d(channel, unsharp_kernel, mode='same', boundary='symm') for channel in im])
    else:
        unsharp = convolve2d(im, unsharp_kernel, mode='same', boundary='symm')
    #im = np.clip(im*(1+weight) - unsharp*weight, 0, np.max(im) if clip_to_max else np.inf)
    im = np.clip(im*(1+weight) - unsharp*weight, 0, np.sum(im)*8/im.size ) ## TODO fix color clipping
    return im
def saturate(im, saturation_enhance):
    monochr = np.dstack([np.sum(im, axis=2)]*3)
    return np.clip(im*(1.+saturation_enhance) - monochr*saturation_enhance, 0, np.max(im))


image_names = sys.argv[1:]

colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len([s for s in sys.argv[1:] if not is_extra(s)])))   ## Generate a nice rainbow scale for all non-extra images
colors = [c*np.array([.8, .7, .9, 1]) for c in colors[::-1]] ## suppress green channel
channel_outputs, extra_outputs = [], []
shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)   ## Initialize affine transform to identity, and image shift to zero
for image_name in image_names:
    ## Load an image
    newimg = safe_imload(image_name)
    color = [1,1,1,1] if is_extra(image_name) else colors.pop()
    print('loading', image_name)
    max_shift = int(rel_max_shift*newimg.shape[0])
    if 'image_padding' not in locals(): image_padding = max_shift*len(sys.argv[1:]) ## temporary very wide black padding for image alignment

    newimg_crop = gaussian_filter(newimg, sigma=decim*.5)[max_shift:-max_shift-int(newimg.shape[0]*databar_pct):decim, max_shift:-max_shift:decim]*1.0

    if 'refimg' in locals(): ## the first image will be simply put to centre (nothing to align against)
        shiftvec_new, trmatrix_new = find_affine_and_shift(refimg_crop, newimg_crop)
        shiftvec_sum, trmatrix_sum = shiftvec_sum + shiftvec_new,  trmatrix_sum + trmatrix_new - np.eye(2)
        print('... is shifted by {:}px against the previous one and by {:}px against the first one'.format(shiftvec_new, shiftvec_sum))
    
    if not is_extra(image_name):
        ## Process the new added image
        im_unsharp = my_affine_tr(np.pad(unsharp_mask(newimg, weight=unsharp_weight, radius=unsharp_radius), pad_width=max_shift, mode='constant'), trmatrix_sum) 
        ## Prepare the composite canvas with the first centered image
        if 'composite_output' not in locals(): composite_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
        paste_overlay(composite_output, im_unsharp, shiftvec_sum, color, normalize=np.max(newimg_crop))

    ## Export an individual channel
    single_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
    if not is_extra(image_name):
        paste_overlay(single_output, im_unsharp, shiftvec_sum, color, normalize=np.max(newimg_crop))
    else:
        paste_overlay(single_output, my_affine_tr(np.pad(newimg, pad_width=max_shift, mode='constant'), trmatrix_sum), shiftvec_sum, color, normalize=np.max(newimg_crop)) 
    (extra_outputs if is_extra(image_name) else channel_outputs).append((single_output,image_name))

    if not consecutive_alignment:   ## optionally, search alignment against the very first image
        shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)
    elif is_extra(image_name):      ## extra imgs never affect alignment of further images
        shiftvec_sum, trmatrix_sum = shiftvec_sum - shiftvec_new,   trmatrix_sum - (trmatrix_new - np.eye(2)) 

    if 'refimg' not in locals() or (consecutive_alignment and not is_extra(image_name)): ## store the image as a reference
        refimg, refimg_crop = newimg, newimg[:-int(newimg.shape[0]*databar_pct):decim, ::decim]*1.0

## Crop all images identically, according to the extent of unused black margins in the composite image
for croppx in range(int(max(composite_output.shape)/2)):
    if not (np.all(composite_output[:,croppx,:] == 0) and np.all(composite_output[:,-croppx,:] == 0) \
            and np.all(composite_output[croppx,:,:] == 0) and np.all(composite_output[-croppx,:,:] == 0)):
        print('can crop', croppx)
        break

## TODO first crop, then annotate each image (separately)
for n,(i,f) in enumerate(channel_outputs): 
    imageio.imsave(str(Path(f).parent / ('channel{:02d}_'.format(n) + Path(f).stem +'.png')), i[croppx:-croppx,croppx:-croppx,:])
for n,(i,f) in enumerate(extra_outputs): 
    imageio.imsave(str(Path(f).parent / ('extra{:02d}_'.format(n) + Path(f).stem.lstrip('+')+ '.png')), i[croppx:-croppx,croppx:-croppx,:])
imageio.imsave(str(Path(f).parent / ('composite_saturate.png')), saturate(composite_output, saturation_enhance=saturation_enhance)[croppx:-croppx,croppx:-croppx,:])
imageio.imsave(str(Path(f).parent / 'composite.png'), composite_output[croppx:-croppx,croppx:-croppx,:])
