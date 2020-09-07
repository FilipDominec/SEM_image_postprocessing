#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""
Multichannel overlay 

Most important features:
    * Automatically aligns images so that their edge features are matched. (Technically, it maximizes abs value of Laplacians.)
    * Optionally, affine transform can be used for better matching images that are rotated or slightly distorted. (This happens
      typically when e.g. electron acceleration voltage change between images or when sample gradually 
      accumulates charge.)
    * Exports the coloured composite image, along with each channel separately. 
    * Additionally, "extra" images can be aligned and exported in grayscale, but they do not affect alignment of further channels. 
      This can be used, e.g., to align SEM images to a series of spectrally resolved cathodoluminescence mappings.

Additional features
    * Images can be sharpened and contrast-enhanced
    * Since electron-microscope images can contain a databar, the bottom portion of the images is ignored during alignment

TODOs: 
    * join the export with annotate.py
    * don't forget to anisotropically scale at the load time
    * (too aggressive clipping, or clipping channels too early?) colors are still not as crisp as they used to be 
    * is there some colour mismatch between affine-tr and normal operation?
    * interactive GUI D&D application
    * test on windows
"""

## User settings

# Settings for correlation of images:
DISABLE_TRANSFORM = False   ## if set to true, the images will just be put atop of each other (no shift, no affine tr.)
USE_AFFINE_TRANSFORM = 0    ## enables scaling, tilting and rotating the images; otherwise they are just shifted
rel_max_shift=.05           ## pixels cropped from the second image determine the maximum shift to be detected (higher number results in slower computation)
DECIM=2                     ## decimation of images prior to correlation (value of 2-5 speeds up processing, but does not affect the results much)
databar_pct = (61./484)     ## relative height of databar at the images' bottom - these are ignored when searching for correlation
#databar_pct =  0.01            ##     (when no databar present)
consecutive_alignment = True ## if disabled, images are aligned always to the first one

EXTRA_IMG_IDENT = 'S'   # each image containing this in its name is treated as extra
EXTRA_IMG_LABEL = '+'   # each image name preceded by this is treated as extra (and this symbol is removed prior to loading)
def is_extra(imname): return (imname[0] == EXTRA_IMG_LABEL or (EXTRA_IMG_IDENT in Path(imname).stem.upper())) ## TODO this should be better defined...

# Image post-processing settings:
SATURATION_ENHANCE = .15
UNSHARP_WEIGHT = 0 #2 #2
UNSHARP_RADIUS = 6




## Import common moduli
import sys, os, time, collections, imageio
import matplotlib, matplotlib.cm ## TODO rm dep
print(matplotlib.cm)
from pathlib import Path 
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import affine_transform, zoom
from scipy.optimize import differential_evolution
np.warnings.filterwarnings('ignore')

import pure_numpy_image_processing as pnip
import annotate_image




image_names = sys.argv[1:]

#colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len([s for s in image_names if not is_extra(s)])))   ## Generate a nice rainbow scale for all non-extra images
#colors = [c*np.array([.8, .7, .9, 1]) for c in colors[::-1]] ## suppress green channel
#colors = pnip.rgb_palette(len([s for s in image_names if not is_extra(s)])
colors = [pnip.hsv_to_rgb(h=h) for  h in np.linspace(1, 1.666, len([s for s in image_names if not is_extra(s)]))] 
colors2 = colors[::-1]
WHITE = [1,1,1]
channel_outputs, extra_outputs = [], []
shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)   ## Initialize affine transform to identity, and image shift to zero
for image_name in image_names:
    print('loading', image_name); 
    newimg = pnip.safe_imload(str(Path(image_name).parent / Path(image_name).name.lstrip(EXTRA_IMG_LABEL)), retouch=True)
    color_tint = WHITE if is_extra(image_name) else colors.pop()
    max_shift = int(rel_max_shift*newimg.shape[0])
    if 'image_padding' not in locals(): image_padding = max_shift*len(image_names) ## temporary very wide black padding for image alignment
    newimg_crop = gaussian_filter(newimg, sigma=DECIM*.5)[max_shift:-max_shift-int(newimg.shape[0]*databar_pct):DECIM, max_shift:-max_shift:DECIM]*1.0

    if 'refimg' in locals() and not DISABLE_TRANSFORM: ## the first image will be simply put to centre (nothing to align against)
        shiftvec_new, trmatrix_new = pnip.find_affine_and_shift(refimg_crop, newimg_crop, use_affine_transform=USE_AFFINE_TRANSFORM)
        shiftvec_sum, trmatrix_sum = shiftvec_sum + shiftvec_new*DECIM,  trmatrix_sum + trmatrix_new - np.eye(2)
        print('... is shifted by {:}px against the previous one and by {:}px against the first one'.format(shiftvec_new*DECIM, shiftvec_sum))

    newimg_processed = pnip.my_affine_tr(        ## Process the new image: sharpening, affine transform, and padding...
            np.pad(pnip.unsharp_mask(newimg, weight=(0 if is_extra(image_name) else UNSHARP_WEIGHT), radius=UNSHARP_RADIUS), 
                    pad_width=max_shift, mode='constant'), trmatrix_sum) 
    
    if not is_extra(image_name):            ## ... then shifting and adding the image to the composite canvas
        if 'composite_output' not in locals(): composite_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
        pnip.paste_overlay(composite_output, newimg_processed, shiftvec_sum, color_tint, normalize=np.max(newimg_crop))

    ## Export the image individually (either as colored channel, or as an extra image)
    single_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
    #pnip.paste_overlay(single_output, newimg_processed, shiftvec_sum, color_tint, normalize=np.max(newimg_crop)) # todo?  normalize to newimg_crop or single_output? or rm it?
    pnip.paste_overlay(single_output, newimg_processed, shiftvec_sum, color_tint, normalize=np.sum(newimg_crop)) # todo?  normalize to newimg_crop or single_output? or rm it?
    (extra_outputs if is_extra(image_name) else channel_outputs).append((single_output, image_name, annotate_image.analyze_header_XL30(image_name)))

    if not consecutive_alignment:   ## optionally, search alignment against the very first image
        shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)
    elif is_extra(image_name):      ## extra imgs never affect alignment of further images
        shiftvec_sum, trmatrix_sum = shiftvec_sum - shiftvec_new*DECIM,   trmatrix_sum - (trmatrix_new - np.eye(2)) 

    if 'refimg' not in locals() or (consecutive_alignment and not is_extra(image_name)): ## store the image as a reference
        refimg, refimg_crop = newimg, newimg[:-int(newimg.shape[0]*databar_pct):DECIM, ::DECIM]*1.0

## Crop all images identically, according to the extent of unused black margins in the composite image # TODO indep cropping on X and Y
for croppx in range(int(max(composite_output.shape)/2)):
    if not (np.all(composite_output[:,croppx,:] == 0) and np.all(composite_output[:,-croppx,:] == 0) \
            and np.all(composite_output[croppx,:,:] == 0) and np.all(composite_output[-croppx,:,:] == 0)):
        #print('can crop', croppx)
        break

for n,(im,f,ih) in enumerate(channel_outputs): 
    im = annotate_image.add_databar_XL30(im[croppx:-croppx,croppx:-croppx,:], f, ih)
    imageio.imsave(str(Path(f).parent / ('channel{:02d}_'.format(n) + Path(f).stem +'_ANNOT2.png')), im)
    
for n,(im,f,ih) in enumerate(extra_outputs): 
    im = annotate_image.add_databar_XL30(im[croppx:-croppx,croppx:-croppx,:], f, ih)
    imageio.imsave(str(Path(f).parent / ('extra{:02d}_'.format(n) + Path(f).stem.lstrip('+')+ '.png')), im)

summary_ih = ih ## TODO: extract lambdas (and, todo later other params) and build coloured list

## Generate 5th line in the databar: color coding explanation
k, vs = annotate_image.extract_dictkey_that_differs(
        [co[2] for co in channel_outputs], key_filter=['Magnification', 'lDetName', 'flAccV'])
if not k:
    k, vs = annotate_image.extract_stringpart_that_differs([co[1] for co in channel_outputs], arbitrary_field_name='λ(nm)')
dbar_appendix = [[[[.6,.6,.6], 'Color coding by '], [WHITE, k+': ' ], ]]
for c, v in zip(colors2, vs): dbar_appendix[0].append([c,' '+v]) ## append to 0th line of the appending
print(dbar_appendix)

composite_output /= np.max(composite_output) # normalize all channels
imageio.imsave(str(Path(f).parent / ('composite_saturate.png')), 
        annotate_image.add_databar_XL30( pnip.saturate(composite_output, saturation_enhance=SATURATION_ENHANCE)[croppx:-croppx,croppx:-croppx,:], f, 
            summary_ih, appendix_lines=dbar_appendix))
imageio.imsave(str(Path(f).parent / 'composite.png'), composite_output[croppx:-croppx,croppx:-croppx,:])
